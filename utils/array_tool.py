import torch as t
import numpy as np

def tonumpy(data):
    if isinstance(data,np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()

def totesnor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

# 标量（0维张量） scalar 只有大小 1，2，3
# 向量（1维张量） vector 大小和方向 （1，2，3）
# 矩阵（2维张量） matrix 好几个向量组合
# 张量 Tensor

# torch.Tensor 和 np.ndarray 可以相互转化
# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)
# tensor_to_array = torch_data.numpy()

# torch.Tensor:是一种包含一种数据类型的的多维矩阵
def scalar(data):
    if isinstance(data, np.ndarray):
        # 从数组中取出数字
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        # 讲标量Tensor转变为Python数字
        # tensor([2.3466])
        # 2.3466
        return data.item()