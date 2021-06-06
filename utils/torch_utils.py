import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)

def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1,0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

