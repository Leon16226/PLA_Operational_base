import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions.weibull import Weibull
from torch.distributions.normal import Normal
from reliability.Fitters import Fit_Weibull_2P
import matplotlib.pyplot as plt



if __name__ == '__main__':
   l1 = torch.tensor([[[1,2,3],[4,5,6]],
                  [[1,2,3],[4,5,6]]])

   l1 = torch.flatten(l1)
   print(l1)



