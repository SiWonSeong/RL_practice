import torch.nn.functional as F

def train(q, q_target, memory, optimizer, batch_size, gamma, device='cpu'):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size) # cpu
        # q, q_target, s, a, s_prime: gpu
        # memory, r, gamma: cpu

        s = s.to(device)
        a = a.to(device)
        s_prime = s_prime.to(device)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1).to('cpu')
        target = r + gamma * max_q_prime * done_mask # cpu
        loss = F.smooth_l1_loss(q_a.to(device), target.to(device))

        optimizer.zero_grad()
        loss.backward() # loss에 대한 그라디언트 계산이 일어남
        optimizer.step() # Qnet의 파라미터 업데이트