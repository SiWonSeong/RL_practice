import torch
import numpy as np
import matplotlib.pyplot as plt

def cosine_fun(x):
    noise = np.random.rand(x.shape[0]) * 0.4 - 0.2 # ε ~ U(0, 1) → ε ~ U(-0.2, 0.2)
    return np.cos(1.5 * np.pi * x) * x + noise # f(x) = cos(1.5 π * x) + x + ε

def plot_results(model):
    x = np.linspace(0, 5, 100)
    input_x = torch.from_numpy(x).float().unsqueeze(1)
    plt.plot(x, cosine_fun(x), label="Truth")
    plt.plot(x, model(input_x).detach().numpy(), label="Prediction")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0, 5))
    plt.ylim((-5, 10))
    plt.grid()
    plt.show()

def pytorch_gpu_check(gpu_num = 0):
    # GPU 인식 여부 확인 및 GPU 할당 변경하기
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu') # 원하는 GPU 번호 사용
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device:', torch.cuda.current_device())  # check
    print(torch.cuda.get_device_name(gpu_num))
    # 사용방법: 명령어.to(device) → GPU환경, CPU환경에 맞춰서 동작
    return device

def pytorch_gpu_allocated_info(gpu_num = 0, unit_num = 3):
    # Additional Infos
    if unit_num == 0:
        unit_str = 'B'
    elif unit_num == 1:
        unit_str = 'KB'
    elif unit_num == 2:
        unit_str = 'MB'
    else:
        unit_num = 3
        unit_str = 'GB'

    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')  # 원하는 GPU 번호 사용
    if device.type == 'cuda':
        print('\tGPU Memory Allocated:', round(torch.cuda.memory_allocated(gpu_num) / 1024 ** unit_num, 1), unit_str)
        print('\tGPU Memory Cached:   ', round(torch.cuda.memory_reserved(gpu_num) / 1024 ** unit_num, 1), unit_str)