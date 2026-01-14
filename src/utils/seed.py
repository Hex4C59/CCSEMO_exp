import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置随机种子，并尽量启用确定性行为."""

    # Python / NumPy / PyTorch 随机数生成器
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN 相关设置：关闭 benchmark，启用 deterministic，减少非确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



