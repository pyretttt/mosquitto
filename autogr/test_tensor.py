import pytest

from tensor import Tensor
import functions as F
import numpy as np
import torch

def _make_data(shape, requires_grad: bool = True):
    X_np = np.random.randn(*shape)
    X_tens = Tensor.from_numpy(data=X_np, requires_grad=requires_grad)
    X_torch = torch.from_numpy(X_np)
    if requires_grad:
        X_torch.requires_grad_(requires_grad)

    return X_np, X_tens, X_torch


def test_tensor_scalar_add_backward():
    X_np, X_tens, X_torch = _make_data((8, 16, 32))
    a = 10
    Y_tens = X_tens + a
    Y_torch = X_torch + a

    w_tens = Y_tens.mean()
    w_torch = torch.mean(Y_torch)
    w_tens.backward()
    w_torch.backward()
    assert np.allclose(X_tens.grad.data, X_torch.grad.numpy())


def test_batched_tensor_mat_mul_backward():
    X_np, X_tens, X_torch = _make_data((8, 16, 32))
    Y_np, Y_tens, Y_torch = _make_data((32, 24))
    Z = X_tens.matmul(Y_tens)
    W = Z.mean()
    W.backward()

    Z_torch = X_torch.matmul(Y_torch)
    W_torch = Z_torch.mean()
    W_torch.backward()

    assert (
        np.allclose(X_tens.grad.data, X_torch.grad.numpy())
        and np.allclose(Y_tens.grad.data, Y_torch.grad.numpy())
    )


def test_sin():
    X_np, X_tens, X_torch = _make_data((8, 16, 32))
    Z = F.sin(X_tens)
    W = Z.mean()
    W.backward()

    Z_torch = torch.sin(X_torch)
    W_torch = Z_torch.mean()
    W_torch.backward()

    assert np.allclose(X_tens.grad.data, X_torch.grad.numpy())


def test_relu():
    X_np, X_tens, X_torch = _make_data((8, 16, 32))
    Z = F.relu(X_tens)
    W = Z.mean()
    W.backward()

    Z_torch = torch.relu(X_torch)
    W_torch = Z_torch.mean()
    W_torch.backward()

    assert np.allclose(X_tens.grad.data, X_torch.grad.numpy())

def test_batched_tensor_mat_mul_with_relu_backward():
    X_np, X_tens, X_torch = _make_data((8, 16, 32))
    Y_np, Y_tens, Y_torch = _make_data((32, 24))
    Z = X_tens.matmul(Y_tens)
    P = F.relu(Z)
    W = P.mean()
    W.backward()

    Z_torch = X_torch.matmul(Y_torch)
    P_torch = torch.relu(Z_torch)
    W_torch = P_torch.mean()
    W_torch.backward()

    assert (
        np.allclose(X_tens.grad.data, X_torch.grad.numpy())
        and np.allclose(Y_tens.grad.data, Y_torch.grad.numpy())
    )