"""
Tensor debugging helpers for PyTorch.

This module monkey-patches several lightweight inspection helpers onto
``torch.Tensor`` to make interactive debugging in notebooks and REPL
sessions faster and less verbose.

The helpers are intentionally short so that inspecting model activations
(e.g., hidden states, logits, attention outputs) becomes ergonomic.

Typical usage:

>>> import torch
>>> import tensor_debug   # patches torch.Tensor

>>> x = torch.randn(2,3,4)

Quick inspection
----------------

>>> x.s
torch.Size([2, 3, 4])

>>> x.nd
3

>>> x.v0.s
torch.Size([4])

"""

import torch


def _t2n(self):
    """
    Convert tensor to NumPy safely.

    Equivalent to ``tensor.detach().cpu().numpy()``.

    >>> import torch
    >>> x = torch.tensor([1.,2.,3.], requires_grad=True)
    >>> x.t2n()
    array([1., 2., 3.], dtype=float32)
    """
    return self.detach().cpu().numpy()


@property
def _s(self):
    """
    Shortcut for tensor shape.

    >>> import torch
    >>> x = torch.zeros(2,3,4)
    >>> x.s
    torch.Size([2, 3, 4])
    """
    return self.shape


@property
def _nd(self):
    """
    Number of tensor dimensions.

    >>> import torch
    >>> torch.zeros(2,3,4).nd
    3
    """
    return self.ndim


@property
def _dt(self):
    """
    Tensor dtype.

    >>> import torch
    >>> torch.zeros(3).dt
    torch.float32
    """
    return self.dtype


@property
def _dev(self):
    """
    Tensor device.

    >>> import torch
    >>> torch.zeros(3).dev
    device(type='cpu')
    """
    return self.device


@property
def _n(self):
    """
    Alias for ``detach().cpu().numpy()``.

    >>> import torch
    >>> x = torch.tensor([1,2,3])
    >>> x.n
    array([1, 2, 3])
    """
    return self.detach().cpu().numpy()


@property
def _v0(self):
    """
    Return the first vector along the last dimension.

    Works for tensors with >=2 dimensions.

    Examples:

    >>> import torch
    >>> x = torch.arange(24).reshape(2,3,4)

    first vector

    >>> x.v0
    tensor([0, 1, 2, 3])

    Equivalent to

    >>> x[0,0,:]
    tensor([0, 1, 2, 3])
    """
    if self.ndim <= 1:
        return self
    return self[(0,) * (self.ndim - 1) + (slice(None),)]


def _v(self, *idx_prefix):
    """
    Return vector at given prefix indices along last dimension.

    Useful for tensors shaped like ``[B, T, D]``.

    >>> import torch
    >>> x = torch.arange(24).reshape(2,3,4)

    Default = first vector

    >>> x.v()
    tensor([0, 1, 2, 3])

    Select specific vector

    >>> x.v(1,2)
    tensor([20, 21, 22, 23])
    """
    if self.ndim <= 1:
        return self

    needed = self.ndim - 1

    if len(idx_prefix) == 0:
        idx_prefix = (0,) * needed

    if len(idx_prefix) != needed:
        raise ValueError(f"Expected {needed} prefix indices")

    return self[idx_prefix + (slice(None),)]


def _same(self, other):
    """
    Exact tensor equality.

    >>> import torch
    >>> a = torch.tensor([1,2,3])
    >>> b = torch.tensor([1,2,3])
    >>> a.same(b)
    True

    >>> b[0] = 9
    >>> a.same(b)
    False
    """
    return torch.equal(self, other)


def _close(self, other, atol=1e-6, rtol=1e-5):
    """
    Floating point approximate equality.

    >>> import torch
    >>> a = torch.tensor([1.0,2.0])
    >>> b = a + 1e-7
    >>> a.close(b)
    True
    """
    return torch.allclose(self, other, atol=atol, rtol=rtol)


@property
def _hasnan(self):
    """
    Check if tensor contains NaN.

    >>> import torch
    >>> x = torch.tensor([1.0, float('nan')])
    >>> x.hasnan
    True
    """
    if not self.is_floating_point():
        return False
    return torch.isnan(self).any().item()


@property
def _hasinf(self):
    """
    Check if tensor contains Inf.

    >>> import torch
    >>> x = torch.tensor([1.0, float('inf')])
    >>> x.hasinf
    True
    """
    if not self.is_floating_point():
        return False
    return torch.isinf(self).any().item()


def _peek(self, k=8):
    """
    Return compact tensor statistics for debugging.

    Includes:
    - shape
    - dtype
    - device
    - mean / std
    - min / max
    - sample values

    >>> import torch
    >>> x = torch.arange(10.)
    >>> stats = x.peek()
    >>> stats["shape"]
    (10,)
    >>> len(stats["sample"]) <= 8
    True
    """
    x = self.detach()

    xf = x.float() if x.is_floating_point() else x

    return {
        "shape": tuple(x.shape),
        "dtype": x.dtype,
        "device": str(x.device),
        "requires_grad": x.requires_grad,
        "hasnan": torch.isnan(x).any().item() if x.is_floating_point() else False,
        "hasinf": torch.isinf(x).any().item() if x.is_floating_point() else False,
        "min": x.min().item() if x.numel() else None,
        "max": x.max().item() if x.numel() else None,
        "mean": xf.mean().item() if x.numel() else None,
        "std": xf.std().item() if x.numel() > 1 else 0.0,
        "sample": x.flatten()[:k].detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------
# Patch torch.Tensor
# ---------------------------------------------------------------------

torch.Tensor.t2n = _t2n
torch.Tensor.s = _s
torch.Tensor.nd = _nd
torch.Tensor.dt = _dt
torch.Tensor.dev = _dev
torch.Tensor.n = _n
torch.Tensor.v0 = _v0
torch.Tensor.v = _v
torch.Tensor.same = _same
torch.Tensor.close = _close
torch.Tensor.hasnan = _hasnan
torch.Tensor.hasinf = _hasinf
torch.Tensor.peek = _peek
