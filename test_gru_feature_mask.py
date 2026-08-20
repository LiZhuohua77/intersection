import unittest

import gymnasium as gym
import numpy as np
import torch

from agent import HybridFeaturesExtractor
from config import (
    AV_OBS_DIM,
    FEATURES_PER_STEP,
    HV_OBS_DIM,
    PREDICTION_HORIZON,
    TOTAL_OBS_DIM,
)


class HybridFeaturesExtractorMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_OBS_DIM,),
            dtype=np.float32,
        )
        self.extractor = HybridFeaturesExtractor(
            observation_space,
            av_obs_dim=AV_OBS_DIM,
            hv_obs_dim=HV_OBS_DIM,
            traj_len=PREDICTION_HORIZON,
            traj_feat_dim=FEATURES_PER_STEP,
            rnn_hidden_dim=64,
        )
        self.base_dim = AV_OBS_DIM + HV_OBS_DIM
        self.traj_dim = PREDICTION_HORIZON * FEATURES_PER_STEP

    def _encoder_grad_sum(self, encoder_name):
        encoder = getattr(self.extractor, encoder_name)
        return sum(
            float(parameter.grad.abs().sum())
            for parameter in encoder.parameters()
            if parameter.grad is not None
        )

    def test_all_zero_padding_has_zero_embedding_and_zero_gru_gradient(self):
        observations = torch.randn(8, TOTAL_OBS_DIM)
        observations[:, self.base_dim :] = 0

        features = self.extractor(observations)
        trajectory_features = features[:, self.base_dim :]
        self.assertTrue(torch.equal(trajectory_features, torch.zeros_like(trajectory_features)))

        features.sum().backward()
        self.assertEqual(self._encoder_grad_sum("yield_traj_encoder"), 0.0)
        self.assertEqual(self._encoder_grad_sum("go_traj_encoder"), 0.0)

    def test_present_and_missing_trajectories_are_masked_independently(self):
        observations = torch.zeros(2, TOTAL_OBS_DIM)
        yield_start = self.base_dim
        go_start = yield_start + self.traj_dim
        observations[0, yield_start : yield_start + self.traj_dim] = 0.25
        observations[1, go_start : go_start + self.traj_dim] = -0.25

        features = self.extractor(observations)
        yield_embedding = features[:, self.base_dim : self.base_dim + 64]
        go_embedding = features[:, self.base_dim + 64 :]

        self.assertGreater(float(yield_embedding[0].abs().sum()), 0.0)
        self.assertEqual(float(yield_embedding[1].abs().sum()), 0.0)
        self.assertEqual(float(go_embedding[0].abs().sum()), 0.0)
        self.assertGreater(float(go_embedding[1].abs().sum()), 0.0)

    def test_incorrect_observation_width_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "expected observations"):
            self.extractor(torch.zeros(2, TOTAL_OBS_DIM - 1))


if __name__ == "__main__":
    unittest.main()
