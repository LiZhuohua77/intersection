import numpy as np

class IntersectionManager:
    """
    定义并查询交叉口的通行优先级规则。
    这是一个无状态的工具类，作为“规则手册”存在。
    """
    def __init__(self):
        # 定义12种基本路径
        self.movements = [
            'S_N', 'S_W', 'S_E', 'N_S', 'N_E', 'N_W',
            'E_W', 'E_S', 'E_N', 'W_E', 'W_N', 'W_S'
        ]
        self.move_map = {move: i for i, move in enumerate(self.movements)}
        
        # 优先级矩阵: P[row, col] = 1 -> row yields to col (行让列)
        # 0: 无冲突, 1: 让行, 2: 优先, 3: 汇入冲突(先到先得)
        self.priority_matrix = np.zeros((12, 12), dtype=int)
        
        self._build_priority_rules()

    def _set_rule(self, move1, move2, rule_type):
        """内部辅助函数，设置双向规则"""
        try:
            idx1 = self.move_map[move1]
            idx2 = self.move_map[move2]
            if rule_type == 'yield':
                self.priority_matrix[idx1, idx2] = 1  # move1 yields to move2
                self.priority_matrix[idx2, idx1] = 2  # move2 has priority over move1
            elif rule_type == 'merge':
                self.priority_matrix[idx1, idx2] = 3
                self.priority_matrix[idx2, idx1] = 3
        except KeyError as e:
            print(f"Warning: Invalid movement key in rule setting: {e}")

    def _build_priority_rules(self):
        """构建所有通行规则"""
        # 规则1: 左转让直行
        self._set_rule('S_W', 'N_S', 'yield') # 南向西(左转) vs 北向南(直行)
        self._set_rule('N_E', 'S_N', 'yield') # 北向东(左转) vs 南向北(直行)
        self._set_rule('E_S', 'W_E', 'yield') # 东向南(左转) vs 西向东(直行)
        self._set_rule('W_N', 'E_W', 'yield') # 西向北(左转) vs 东向西(直行)

        # 规则2: 左转让对向右转 (本质是让行对向车流)
        self._set_rule('S_W', 'N_W', 'yield') # 南向西(左) vs 北向西(右)
        self._set_rule('N_E', 'S_E', 'yield') # 北向东(左) vs 南向东(右)
        self._set_rule('E_S', 'W_S', 'yield') # 东向南(左) vs 西向南(右)
        self._set_rule('W_N', 'E_N', 'yield') # 西向北(左) vs 东向北(右)
        
        # 规则3: 直行与右转的冲突 (直行有更高优先级)
        self._set_rule('W_S', 'S_N', 'yield') # 西向南(右) vs 南向北(直)
        self._set_rule('E_N', 'N_S', 'yield') # 东向北(右) vs 北向南(直)
        # ... 可以根据需要添加更多规则

        # 规则4: 无法直接判断优先级的，设为汇入冲突，靠先到先得解决
        self._set_rule('S_N', 'W_E', 'merge') # 直行 vs 右侧直行

    def check_yield_required(self, ego_move, other_move):
        """
        检查ego路径是否需要让行other路径。
        Returns:
            int: 1 (必须让行), 3 (汇入冲突，需进一步判断), 0 (其他情况)
        """
        if ego_move not in self.move_map or other_move not in self.move_map:
            return 0
        
        priority_val = self.priority_matrix[self.move_map[ego_move], self.move_map[other_move]]
        return priority_val if priority_val in [1, 3] else 0