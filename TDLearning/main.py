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
    alpha = 0.01  # MC에 비해 큰 값을 사용

    for k in range(50000):  # 총 5만 번의 에피소드 진행
        done = False
        while not done:
            y, x = env.get_state()
            action = agent.select_action()
            (_, _), reward, done = env.step(action) # (y_prime, x_prime), reward, done
            y_prime, x_prime = env.get_state()

            # 한 번의 step이 진행되자 마자 바로 테이블의 데이터를 업데이트 해줌
            data[y][x] = data[y][x] + alpha * (reward + gamma * data[y_prime][x_prime] - data[y][x])
        env.reset()

    # 학습이 끝나고 난 후 데이터를 출력해보기 위한 코드
    for row in data:
        print(row)


if __name__ == '__main__':
    main()
