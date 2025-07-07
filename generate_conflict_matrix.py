# generate_conflict_matrix.py
import pprint

def generate_conflict_matrix():
    """
    通过预定义的逻辑规则，生成交叉口路径冲突矩阵。
    """
    maneuvers = [
        'N_S', 'N_W', 'N_E', 'S_N', 'S_E', 'S_W',
        'E_W', 'E_N', 'E_S', 'W_E', 'W_S', 'W_N'
    ]

    opposites = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    rot_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
    rot_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}

    def get_attrs(m):
        start, end = m.split('_')

        if end == opposites[start]: maneuver_type = 'straight'
        elif end == rot_right[start]: maneuver_type = 'right'
        else: maneuver_type = 'left'
        
        return {'start': start, 'end': end, 'type': maneuver_type}

    matrix = {m: {n: False for n in maneuvers} for m in maneuvers}

    for m1 in maneuvers:
        for m2 in maneuvers:
            if m1 == m2:
                continue

            attr1 = get_attrs(m1)
            attr2 = get_attrs(m2)

            # 如果已经计算过对称项，则跳过
            if matrix[m2][m1]:
                matrix[m1][m2] = True
                continue

            # 规则1：起点相同，无冲突
            if attr1['start'] == attr2['start']:
                is_conflict = False
            # 规则2：起点和终点互为相反（例如N->S 和 S->N），即对向行驶
            elif attr1['start'] == attr2['end'] and attr1['end'] == attr2['start']:
                is_conflict = (attr1['type'] == 'left' or attr2['type'] == 'left')
            else:
                is_conflict = True

                if attr1['type'] == 'right':
                    if get_attrs(m2)['start'] == rot_left[attr1['start']] and attr2['type'] != 'left':
                        is_conflict = False

                if attr2['type'] == 'right':
                    if get_attrs(m1)['start'] == rot_left[attr2['start']] and attr1['type'] != 'left':
                        is_conflict = False

                if (m1, m2) in [('N_W', 'E_S'), ('E_S', 'N_W'), 
                               ('W_S', 'S_E'), ('S_E', 'W_S'), 
                               ('S_E', 'E_N'), ('E_N', 'S_E'),
                               ('E_N', 'N_W'), ('N_W', 'E_N')]:
                    is_conflict = True

            matrix[m1][m2] = is_conflict

    return matrix


if __name__ == '__main__':
    conflict_matrix = generate_conflict_matrix()
    print("Pre-computed Conflict Matrix:")
    pprint.pprint(conflict_matrix)