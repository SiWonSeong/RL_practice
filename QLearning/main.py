from GridWorld import *
from QAgent import *

def main():
    env = GridWorld()
    agent = QAgent()
    for n_episode in range(10000):  # 총 10000 에피소드 동안 학습
        done = False

        s = env.reset()
        while not done:  # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = s_prime
        agent.anneal_eps()

    agent.show_table()  # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()
