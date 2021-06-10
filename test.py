import numpy as np
import torch

if __name__ == "__main__":
    p1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    p2 = torch.tensor([[0.0, 0.0], [-1.0, -1.0]])
    p = torch.cdist(p1, p2, p=1.0)
    print(p)

