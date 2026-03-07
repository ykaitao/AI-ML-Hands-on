import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch


def _t2n(self):
    """Convert a PyTorch tensor to a NumPy array.
    >>> _ = torch.manual_seed(42)
    >>> x = torch.rand(2,2, requires_grad=True)
    >>> x
    tensor([[0.8823, 0.9150],
            [0.3829, 0.9593]], requires_grad=True)
    >>> x.t2n() # x.detach().numpy()
    array([[0.88226926, 0.91500396],
           [0.38286376, 0.95930564]], dtype=float32)

    Calling ``.numpy()`` directly on a tensor that requires grad raises an error:

    >>> x.cpu().numpy()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: Can't call numpy() on Tensor that requires grad...
    """
    # detach so we don’t keep the grad graph and then convert to NumPy
    return self.cpu().detach().numpy()


# monkey‑patch the method onto the Tensor class
torch.Tensor.t2n = _t2n
