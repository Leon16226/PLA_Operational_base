import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def argmax():
    x = np.array([-2, 3, -9.4, 5, 0])
    ex = np.sum(np.exp(x) / np.sum(np.exp(x)) * np.array([0, 1, 2, 3, 4]))
    print(ex)


# 可微
def softargmax():
    x = np.array([-2, 3, -9.4, 5, 0])
    x = x * 10  # beta=10
    ex = np.sum(np.exp(x) / np.sum(np.exp(x)) * np.array([0, 1, 2, 3, 4]))
    print(ex)


# 和为1
# 在分类任务中，作为概率出现在交叉地损失函数中
def softmax():
    x = np.array([0.5, 2, 1.4, 3, 9.1, -1])
    softmax = np.exp(x) / np.sum(np.exp(x))
    print(np.sum(softmax))


# 加log解决上下溢出问题
def logsoftmax():
    x = torch.Tensor([-4, 2, -3.2, 0, 7])
    softmax = torch.exp(x) / torch.sum(torch.exp(x))
    logsoftmax = torch.log(softmax)
    print(torch.sum(logsoftmax))

# -log(softmax())
# 是NCE the Noise Contrastive Estimation loss
# 噪声对比估计损失函数

# the representations obtained place similar images closer in the space

# 负对数似然损失函数
def NLLLoss():
    x = torch.randn(5, 5)
    target = torch.tensor([0, 2, 3, 1, 4])
    # one-hot
    one_hot = F.one_hot(target).float()
    softmax = torch.exp(x) / torch.sum(torch.exp(x), dim=1).reshape(-1, 1)
    logsoftmax = torch.log(softmax)
    nllloss = -torch.sum(one_hot * logsoftmax) / target.shape[0]
    print(nllloss)


# cross_entropy_loss = log_softmax + nll_loss
# nll_loss(log_softmax(input, 1), target)
# 训练的是1D张量，做分类
# 交叉熵主要是用来判定实际的输出与期望的输出的接近程度

if __name__ == "__main__":
    argmax()
    softargmax()
    softmax()
