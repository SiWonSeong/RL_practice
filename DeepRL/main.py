import torch.optim as optim
import time
from DeepRL.Model import *
from utils import *

def main():
    # PyTorch 버전 확인
    print("torch version:", torch.__version__)

    # GPU 인식 여부 확인 및 GPU 할당 변경
    gpu_num = 0
    device = pytorch_gpu_check(gpu_num = gpu_num)
    print('device:', device)

    # 0~5 사이 숫자 1만개를 샘플링하여 인풋으로 사용
    data_x = np.random.rand(10000) * 5

    model = Model().to(device)
    before_train_model = Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epoch = 10000
    print_interval = 1000
    start_time = time.time()
    for step in range(1, epoch+1):
        batch_x = np.random.choice(data_x, 32)
        # 랜덤하게 뽑힌 32개의 데이터로 mini-batch를 구성
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1).to('cpu')
        pred = model(batch_x_tensor.to(device))

        batch_y = cosine_fun(batch_x)
        truth = torch.from_numpy(batch_y).float().unsqueeze(1).to('cpu')
        loss = F.mse_loss(pred, truth.to(device)) # 손실 함수인 MSE를 계산하는 부분

        optimizer.zero_grad()
        loss.mean().backward() # 역전파를 통한 그라디언트 계산이 일어나는 부분
        optimizer.step() # 실제로 파라미터를 업데이트 하는 부분

        if (step % print_interval == 0 and step != 0) or (step == epoch):
            print("\n[%d/%d] loss: %.3f " %(step, epoch, loss.cpu().data.numpy()))
            end_time = time.time()
            print("total time: %.2f sec.." %(end_time - start_time))
            pytorch_gpu_allocated_info(gpu_num = gpu_num, unit_num=2)

    plot_results(before_train_model.to('cpu'))
    plot_results(model.to('cpu'))

if __name__ == '__main__':
    main()
