import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
1.1
# 如果 CUDA 可用，会打印出您的 GPU 信息
if torch.cuda.is_available():
    print("当前使用的 GPU 名称:", torch.cuda.get_device_name(0))

# 简单测试一个张量
x = torch.rand(5, 3)
print("测试张量:", x)
