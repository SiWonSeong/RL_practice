import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, learning_rate):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.0])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
            torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list),\
            torch.tensor(r_list, dtype=torch.float), torch.tensor(s_prime_list, dtype=torch.float),\
            torch.tensor(done_list, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self, gamma, device='cpu'):
        s, a, r, s_prime, done = self.make_batch() # cpu

        # s, a, r, s_prime, done, gamma: gpu
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done = done.to(device)

        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach()) # detach: 상수 취급

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()