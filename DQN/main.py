import gym
import time
import torch.optim as optim

from DQN.ReplayBuffer import *
from DQN.Qnet import *
from DQN.train import *
from utils import *

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
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
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # q_target은 업데이트 하지 않음
    start_time = time.time()
    for n_epi in range(1, epoch+1):
        epsilon = max(0.01, 0.08 - 0.01 * ((n_epi-1) / 200)) # Linear annealing from 8% to 1%
        s = env.reset() # numpy
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).to(device).float(), epsilon=epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q=q, q_target=q_target,
                  memory=memory, optimizer=optimizer, batch_size=batch_size, gamma=gamma, device=device)

        if (n_epi % print_interval == 0 and n_epi != 0) or (n_epi == epoch):
            q_target.load_state_dict(q.to(device).state_dict())
            print("\nn_episode: {}, score: {:.1f}, n_buffer: {}, eps: {:.1f}%"
                  .format (n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

            end_time = time.time()
            print("total time: {:.2f} sec.." .format(end_time - start_time))
            pytorch_gpu_allocated_info(gpu_num=gpu_num, unit_num=2)
    env.close()

if __name__ == '__main__':
    main()
