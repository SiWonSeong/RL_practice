import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, learning_rate):
        super(Policy, self).__init__()
        self.data  = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, gamma, device='cpu'):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -R * torch.log(prob.to(device)) # prob: π_θ(s_t, a_t)
            loss.backward()
        self.optimizer.step() # 파라미터 업데이트
        self.data = []

    def train_net_with_out_log(self, gamma, device='cpu'):
        # loss에서 log항을 제거하면 policy gradient theorem에서의 등식이 성립하지 않게 되어 학습이 잘 안된다. (Don't Use!)
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -R * prob.to(device)  # with out log term
            loss.backward()
        self.optimizer.step()  # 파라미터 업데이트
        self.data = []
