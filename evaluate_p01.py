"""Batch evaluation for the P0-1 multi-seed constrained-RL experiments.

This script discovers one final model for every requested (seed, algorithm)
pair, evaluates all models on the same deterministic sequence of environment
seeds, and writes episode-, model-, and algorithm-level CSV summaries.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import MAX_ACCELERATION
from ppo_lagrangian import PPOLagrangian
from sagi_ppo import SAGIPPO
from traffic_env import TrafficEnv


DEFAULT_ALGOS = [
    "sagi_ppo_mlp",
    "sagi_ppo_gru",
    "ppo_lagrangian_mlp",
    "ppo_lagrangian_gru",
]
DEFAULT_SEEDS = [0, 42, 123, 1337, 3407]


@dataclass(frozen=True)
class EvaluationTask:
    train_seed: int
    algo: str
    model_path: str
    scenario: str
    num_episodes: int
    eval_seed: int
    cost_budget: float
    gamma_override: float | None
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all P0-1 final models with one paired protocol."
    )
    parser.add_argument("--model-root", default="models/p0_1_retrain")
    parser.add_argument(
        "--scenario",
        default="agent_only_simple",
        choices=[
            "agent_only_simple",
            "crossing_conflict",
            "random_traffic",
            "mixed_traffic",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--algos", nargs="+", choices=DEFAULT_ALGOS, default=DEFAULT_ALGOS)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=8491)
    parser.add_argument("--cost-budget", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", default="evaluation_results/p0_1")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_tasks(args: argparse.Namespace) -> list[EvaluationTask]:
    model_root = Path(args.model_root).resolve()
    tasks: list[EvaluationTask] = []
    errors: list[str] = []

    for train_seed in args.seeds:
        for algo in args.algos:
            pattern = (
                f"seed_{train_seed}/{args.scenario}/"
                f"{args.scenario}_{algo}_*/{algo}_final_model.zip"
            )
            matches = sorted(model_root.glob(pattern))
            if len(matches) != 1:
                errors.append(
                    f"seed={train_seed}, algo={algo}: expected one model, "
                    f"found {len(matches)} for {pattern}"
                )
                continue
            tasks.append(
                EvaluationTask(
                    train_seed=train_seed,
                    algo=algo,
                    model_path=str(matches[0]),
                    scenario=args.scenario,
                    num_episodes=args.num_episodes,
                    eval_seed=args.eval_seed,
                    cost_budget=args.cost_budget,
                    gamma_override=args.gamma,
                    device=args.device,
                )
            )

    if errors:
        raise RuntimeError("Model discovery failed:\n" + "\n".join(errors))
    return tasks


def calculate_episode_metrics(info: dict[str, Any], env: TrafficEnv) -> dict[str, float]:
    episode_log = info.get("episode_log") or []
    if not episode_log:
        return {
            "avg_jerk_mps3": 0.0,
            "avg_abs_scte_m": 0.0,
            "signed_mean_scte_m": 0.0,
            "energy_net_kwh": 0.0,
            "energy_consumed_kwh": 0.0,
            "energy_regenerated_kwh": 0.0,
        }

    avg_jerk = 0.0
    if "action_accel" in episode_log[0] and len(episode_log) > 1:
        accelerations = np.asarray(
            [row.get("action_accel", 0.0) for row in episode_log], dtype=np.float64
        ) * MAX_ACCELERATION
        avg_jerk = float(np.mean(np.abs(np.diff(accelerations) / env.dt)))

    scte = np.asarray(
        [row.get("signed_cross_track_error", 0.0) for row in episode_log],
        dtype=np.float64,
    )
    avg_abs_scte = float(np.mean(np.abs(scte)))
    signed_mean_scte = float(np.mean(scte))

    power_kw = np.asarray(
        [row.get("raw_power", 0.0) for row in episode_log], dtype=np.float64
    )
    dt_hours = env.dt / 3600.0
    consumed = float(np.sum(np.clip(power_kw, 0.0, None)) * dt_hours)
    regenerated = float(-np.sum(np.clip(power_kw, None, 0.0)) * dt_hours)

    return {
        "avg_jerk_mps3": avg_jerk,
        "avg_abs_scte_m": avg_abs_scte,
        "signed_mean_scte_m": signed_mean_scte,
        "energy_net_kwh": consumed - regenerated,
        "energy_consumed_kwh": consumed,
        "energy_regenerated_kwh": regenerated,
    }


def classify_outcome(info: dict[str, Any], truncated: bool) -> str:
    failure = info.get("failure")
    if failure == "collision":
        return "collision"
    if failure == "off_track":
        return "off_track"
    if truncated:
        return "timeout"
    return "success"


def evaluate_task(task: EvaluationTask) -> list[dict[str, Any]]:
    set_seed(task.eval_seed)
    model_class = PPOLagrangian if task.algo.startswith("ppo_lagrangian") else SAGIPPO
    rows: list[dict[str, Any]] = []

    # The environment prints every reset and episode termination. Suppress those
    # messages so a 2,000-episode batch evaluation keeps a readable console log.
    with open(os.devnull, "w", encoding="utf-8") as null_output:
        with contextlib.redirect_stdout(null_output):
            env = TrafficEnv(scenario=task.scenario)
            model = model_class.load(task.model_path, env=env, device=task.device)
            gamma = float(
                task.gamma_override if task.gamma_override is not None else model.gamma
            )

            for episode in range(1, task.num_episodes + 1):
                episode_seed = task.eval_seed + episode
                set_seed(episode_seed)
                state, _ = env.reset(
                    seed=episode_seed,
                    options={
                        "scenario": task.scenario,
                        # Deliberately omit ``algo``. Training used the common
                        # reward definition and evaluation must do the same.
                    },
                )
                reward_sum = 0.0
                raw_cost = 0.0
                discounted_cost = 0.0
                episode_length = 0

                while True:
                    action, _ = model.predict(state, deterministic=True)
                    state, reward, terminated, truncated, info = env.step(action)
                    step_cost = float(info.get("cost", 0.0))
                    reward_sum += float(reward)
                    raw_cost += step_cost
                    discounted_cost += (gamma**episode_length) * step_cost
                    episode_length += 1

                    if terminated or truncated:
                        row = {
                            "train_seed": task.train_seed,
                            "algo": task.algo,
                            "model_path": task.model_path,
                            "eval_episode": episode,
                            "eval_seed": episode_seed,
                            "reward": reward_sum,
                            "raw_cost": raw_cost,
                            "discounted_cost": discounted_cost,
                            "cost_surplus": discounted_cost - task.cost_budget,
                            "budget_satisfied": int(
                                discounted_cost <= task.cost_budget
                            ),
                            "episode_length": episode_length,
                            "travel_time_s": episode_length * env.dt,
                            "outcome": classify_outcome(info, truncated),
                        }
                        row.update(calculate_episode_metrics(info, env))
                        rows.append(row)
                        break
            env.close()
    return rows


def mean_and_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    ddof = 1 if len(array) > 1 else 0
    return float(np.mean(array)), float(np.std(array, ddof=ddof))


def summarize_model(rows: list[dict[str, Any]], cost_budget: float) -> dict[str, Any]:
    first = rows[0]
    summary: dict[str, Any] = {
        "train_seed": first["train_seed"],
        "algo": first["algo"],
        "model_path": first["model_path"],
        "num_episodes": len(rows),
    }
    continuous_metrics = [
        "reward",
        "raw_cost",
        "discounted_cost",
        "episode_length",
        "travel_time_s",
        "avg_jerk_mps3",
        "avg_abs_scte_m",
        "signed_mean_scte_m",
        "energy_net_kwh",
        "energy_consumed_kwh",
        "energy_regenerated_kwh",
    ]
    for metric in continuous_metrics:
        mean, std = mean_and_std([float(row[metric]) for row in rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

    outcomes = [str(row["outcome"]) for row in rows]
    for outcome in ["success", "collision", "off_track", "timeout"]:
        summary[f"{outcome}_rate"] = outcomes.count(outcome) / len(outcomes)
    summary["episode_budget_satisfied_rate"] = float(
        np.mean([int(row["budget_satisfied"]) for row in rows])
    )
    summary["mean_cost_surplus"] = summary["discounted_cost_mean"] - cost_budget
    summary["mean_cost_constraint_satisfied"] = int(
        summary["discounted_cost_mean"] <= cost_budget
    )
    return summary


def summarize_algorithm(model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    algos = sorted({str(row["algo"]) for row in model_summaries})
    metrics = [
        "reward_mean",
        "discounted_cost_mean",
        "episode_budget_satisfied_rate",
        "success_rate",
        "collision_rate",
        "off_track_rate",
        "timeout_rate",
        "travel_time_s_mean",
        "avg_jerk_mps3_mean",
        "avg_abs_scte_m_mean",
        "energy_net_kwh_mean",
    ]
    for algo in algos:
        rows = [row for row in model_summaries if row["algo"] == algo]
        summary: dict[str, Any] = {
            "algo": algo,
            "num_training_seeds": len(rows),
            "feasible_seed_count": sum(
                int(row["mean_cost_constraint_satisfied"]) for row in rows
            ),
        }
        for metric in metrics:
            mean, std = mean_and_std([float(row[metric]) for row in rows])
            summary[f"{metric}_seed_mean"] = mean
            summary[f"{metric}_seed_std"] = std
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tasks = discover_tasks(args)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root).resolve() / f"{args.scenario}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as output:
        json.dump(vars(args), output, ensure_ascii=False, indent=2)

    started = time.time()
    results: list[list[dict[str, Any]]] = []
    if args.workers == 1:
        for index, task in enumerate(tasks, start=1):
            rows = evaluate_task(task)
            results.append(rows)
            print(
                f"[{index:02d}/{len(tasks)}] seed={task.train_seed} "
                f"algo={task.algo} completed in {time.time() - started:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                results.append(future.result())
                print(
                    f"[{index:02d}/{len(tasks)}] seed={task.train_seed} "
                    f"algo={task.algo} completed in {time.time() - started:.1f}s",
                    flush=True,
                )

    episode_rows = sorted(
        [row for result in results for row in result],
        key=lambda row: (int(row["train_seed"]), str(row["algo"]), int(row["eval_episode"])),
    )
    model_summaries = [
        summarize_model(result, args.cost_budget) for result in results
    ]
    model_summaries.sort(key=lambda row: (int(row["train_seed"]), str(row["algo"])))
    algorithm_summaries = summarize_algorithm(model_summaries)

    write_csv(output_dir / "episode_results.csv", episode_rows)
    write_csv(output_dir / "model_summary.csv", model_summaries)
    write_csv(output_dir / "algorithm_summary.csv", algorithm_summaries)
    print(f"Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
