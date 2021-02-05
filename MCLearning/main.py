from GridWorld import *
from Agent import *

# import A → A라는 이름의 모듈(.py)을 불러온다.
# from A import B → A라는 이름의 모듈(.py)에서 B라는 모듈, 함수, 클래스, 상수 등을 불러온다.
# from 참조할파일(모듈) import *

def main():
    env = GridWorld()
    agent = Agent()
    data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 테이블 초기화
    gamma = 1.0
    alpha = 0.001

    for k in range(50000):  # 총 5만 번의 에피소드 진행
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (y, x), reward, done = env.step(action)
            history.append((y, x, reward))
        env.reset()

        # 매 에피소드가 끝나고 바로 해당 데이터를 이용해 테이블을 업데이트
        cum_reward = 0  # 리턴 G_t
        for transition in history[::-1]:
            # 방문했던 상태들을 뒤에서부터 보며 차례차례 리턴을 계산
            y, x, reward = transition
            data[y][x] = data[y][x] + alpha * (cum_reward - data[y][x])
            cum_reward = reward + gamma * cum_reward

    # 학습이 끝나고 난 후 데이터를 출력해보기 위한 코드
    for row in data:
        print(row)

if __name__ == '__main__':
    main()
