import gym
import time
from torch.distributions import Categorical

from Policy import *
from utils import *

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
epoch = 1000

def main():
    # PyTorch 버전 확인
    print("torch version:", torch.__version__)

    # GPU 인식 여부 확인 및 GPU 할당 변경
    gpu_num = 0
    device = pytorch_gpu_check(gpu_num=gpu_num)
    print('device:', device)

    # create environment
    env = gym.make('CartPole-v1')
    pi = Policy(learning_rate).to(device)

    score = 0.0
    print_interval = 20
    start_time = time.time()
    for n_epi in range(1, epoch+1):
        s = env.reset()
        done = False

        while not done:
            prob = pi(torch.from_numpy(s).to(device).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net(gamma=gamma, device=device)

        if (n_epi % print_interval == 0 and n_epi != 0) or (n_epi == epoch):
            print("\nn_episode: {}, avg score: {}".format(n_epi, score/print_interval))
            score = 0.0

            end_time = time.time()
            print("total time: {:.2f} sec.." .format(end_time - start_time))
            pytorch_gpu_allocated_info(gpu_num=gpu_num, unit_num=2)
    env.close()

if __name__ == '__main__':
    main()
