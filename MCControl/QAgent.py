import random
import numpy as np


class QAgent:
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))  # q-value를 저장하는 변수, 모두 0으로 초기화 - y, x, action
        self.epsilon = 0.9  # ε
        self.alpha = 0.01  # α

    def select_action(self, s):
        # ε-greedy로 액션을 선택해준다.
        y, x = s
        coin = random.random()  # 0 ~ 1 사이의 랜덤 실수
        if coin < self.epsilon:  # epsilon 값의 확률로 새로운 action을 탐색(exploration)
            action = random.randint(0, 3)  # 0 ~ 3 사이의 랜덤 정수 → 랜덤 action
        else:
            action_val = self.q_table[y, x, :]  # Q(s, a)
            action = np.argmax(action_val)  # action = argmax_a( Q(s, a) ) → exploit
        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q_table의 값을 업데이트 한다.
        cum_reward = 0  # reward의 총합(누적)
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            y, x = s
            # 몬테카를로 방식을 이용하여 업데이트 → Q(s, a) = Q(s, a) + α * {r - Q(s, a)}
            self.q_table[y, x, a] = self.q_table[y, x, a] + self.alpha * (cum_reward - self.q_table[y, x, a])
            cum_reward = cum_reward + r

    def anneal_eps(self):
        self.epsilon -= 0.03
        self.epsilon = max(self.epsilon, 0.1)  # epslion을 0.1까지 감소시킴

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_list = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(type(data), data.shape)
        print(data)
        for y in range(data.shape[0]):
            print()
            for x in range(data.shape[1]):
                if (x == 2 and y in [0, 1, 2]) or (x == 4 and y in [2, 3, 4]):
                    print('■', end=' ')
                elif data[y, x] == 0:
                    print('→', end=' ')
                elif data[y, x] == 1:
                    print('←', end=' ')
                elif data[y, x] == 2:
                    print('↑', end=' ')
                elif data[y, x] == 3:
                    print('↓', end=' ')
                else:
                    print(' ', end=' ')
        print()
