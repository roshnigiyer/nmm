import torch
import math


def logmap0(x: torch.Tensor, k=1, dim: int = -1):
    x_norm = x.norm(dim=dim, p=2, keepdim=True)
    return (x / (math.sqrt(k) * x_norm)) * torch.arctanh(math.sqrt(k) * x_norm)


def expmap0(x: torch.Tensor, k=1, dim: int = -1):
    x_norm = x.norm(dim=dim, p=2, keepdim=True)
    return (x / (math.sqrt(k) * x_norm)) * torch.tanh(math.sqrt(k) * x_norm)


def logmap0(x: torch.Tensor, k=1, dim: int = -1):
    x_norm = x.norm(dim=dim, p=2, keepdim=True)
    return (x / (math.sqrt(k) * x_norm)) * torch.arctanh(math.sqrt(k) * x_norm)


def spherical_norm(u, eps=1e-5):
    u_norm = torch.nn.l2_normalize(u, 1)
    u = torch.subtract(torch.divide(u, u_norm), eps)
    return u

def sph_inner(x, u, v, keepdims=False):
    return torch.reduce_sum(u * v, axis=-1, keepdims=keepdims)

def spherical_dist(u,v, keepdims=False):
    # matrices u, v
    # return tf.math.acos(tf.reduce_sum(tf.multiply(u,v), 1))
    inner = sph_inner(u, u, v, keepdims=keepdims)
    cos_angle = torch.clip_by_value(inner, -1.0, 1.0)
    return torch.math.acos(cos_angle)


def spherical_add(u, v, eps=1e-5):

    ret_sum = torch.add(logmap0(expmap0(u,k=0.5), k=0.5), logmap0(expmap0(v,k=0.5),k=0.5))

    # ret_sum_norm = tf.nn.l2_normalize(ret_sum, 1)
    ret_sum_norm = torch.norm(ret_sum, ord=2, axis=None)
    if (ret_sum_norm >= 1):
        ret_sum = torch.subtract(torch.divide(ret_sum, ret_sum_norm), eps)
    return ret_sum


def spherical_matrix_vec_mult(M, u, eps=1e-5):
    ret_mult = torch.matmul(M, logmap0(expmap0(u,k=0.5), k=0.5))
    # ret_mult_norm = tf.nn.l2_normalize(ret_mult, 1)
    ret_mult_norm = torch.norm(ret_mult)
    if (ret_mult_norm >= 1):
        ret_mult = torch.subtract(torch.divide(ret_mult, ret_mult_norm), eps)
    return ret_mult


def spherical_scalar_vec_mult(c, u, eps=1e-5):
    ret_scal_mult = torch.multiply(c,logmap0(expmap0(u,k=0.5), k=0.5))
    # ret_scal_mult_norm = tf.nn.l2_normalize(ret_scal_mult, 1)
    ret_scal_mult_norm = torch.norm(ret_scal_mult, ord=2, axis=None)
    if (ret_scal_mult_norm >= 1):
        ret_scal_mult = torch.subtract(torch.divide(ret_scal_mult, ret_scal_mult_norm), eps)
    return ret_scal_mult


def spherical_vec_vec_mult(u, v, eps=1e-5):
    ret_vec_mult = torch.multiply(logmap0(expmap0(u,k=0.5), k=0.5), logmap0(expmap0(v,k=0.5), k=0.5))
    # ret_vec_mult_norm = tf.nn.l2_normalize(ret_vec_mult, 1)
    ret_vec_mult_norm = torch.norm(ret_vec_mult, ord=2, axis=None)
    if (ret_vec_mult_norm >= 1):
        ret_vec_mult = torch.subtract(torch.divide(ret_vec_mult, ret_vec_mult_norm), eps)
    return ret_vec_mult