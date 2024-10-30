import torch
import math
import numpy as np


def logmap0(x: torch.Tensor, k=-1, dim: int = -1):
    x_norm = x.norm(dim=dim, p=2, keepdim=True)
    return (x / (-1 * math.sqrt(abs(k)) * x_norm)) * torch.arctanh(-1 * math.sqrt(abs(k)) * x_norm)


def expmap0(x: torch.Tensor, k=-1, dim: int = -1):
    x_norm = x.norm(dim=dim, p=2, keepdim=True)
    return (x / (-1 * math.sqrt(abs(k)) * x_norm)) * torch.tanh(-1 * math.sqrt(abs(k)) * x_norm)


# activation
def hyper_norm(u, eps=1e-5):
    u_norm = torch.nn.l2_normalize(u, 1)
    if (u_norm >= 1):
        u = torch.subtract(torch.math.divide(u, u_norm), eps)
    return u

def get_eps(val):
    return np.finfo(val.dtype.name).eps

def hyper_dist(x, y, k=1.0, keepdims=False):
    sqrt_k = torch.math.sqrt(torch.cast(k, x.dtype))
    x_y = hyper_add(-x, y)
    norm_x_y = torch.linalg.norm(x_y, axis=-1, keepdims=keepdims)
    eps = get_eps(x)
    tanh = torch.clip_by_value(sqrt_k * norm_x_y, -1.0 + eps, 1.0 - eps)
    return 2 * torch.math.atanh(tanh) / sqrt_k

def hyper_add(u, v, k=1.0):
    """Compute the Möbius addition of :math:`x` and :math:`y` in
    :math:`\mathcal{D}^{n}_{k}`
    :math:`x \oplus y = \frac{(1 + 2k\langle x, y\rangle + k||y||^2)x + (1
        - k||x||^2)y}{1 + 2k\langle x,y\rangle + k^2||x||^2||y||^2}`
    """
    x_2 = torch.reduce_sum(torch.math.square(u), axis=-1, keepdims=True)
    y_2 = torch.reduce_sum(torch.math.square(v), axis=-1, keepdims=True)
    x_y = torch.reduce_sum(u * v, axis=-1, keepdims=True)
    k = torch.cast(k, u.dtype)
    return ((1 + 2 * k * x_y + k * y_2) * v + (1 - k * x_2) * v) / (
            1 + 2 * k * x_y + k ** 2 * x_2 * y_2
    )

def hyper_matrix_vec_mult(M, u):
    return expmap0(torch.matmul(M, logmap0(u)))

def hyper_scalar_vec_mult(r, x, k=1.0):
    """Compute the Möbius scalar multiplication of :math:`x \in
            \mathcal{D}^{n}_{k} \ {0}` by :math:`r`
            :math:`x \otimes r = (1/\sqrt{k})\tanh(r
            \atanh(\sqrt{k}||x||))\frac{x}{||x||}`
            """
    sqrt_k = torch.math.sqrt(torch.cast(k, x.dtype))
    norm_x = torch.linalg.norm(x, axis=-1, keepdims=True)
    eps = get_eps(x)
    tan = torch.clip_by_value(sqrt_k * norm_x, -1.0 + eps, 1.0 - eps)
    return (1 / sqrt_k) * torch.math.tanh(r * torch.math.atanh(tan)) * x / norm_x

def hyper_vec_vec_mult(u, v):
    temp_mul = torch.multiply(u,v)
    return temp_mul

