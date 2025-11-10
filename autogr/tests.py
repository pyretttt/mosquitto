from tensor import Tensor
import numpy as np

if __name__ == "__main__":
    x = Tensor.from_numpy(np.array([1, 2, 3]).reshape(3, 1), requires_grad=True)
    y = Tensor.from_numpy(np.array([4, 5, 6]).reshape(3, 1), requires_grad=True)
    z = x + y
    w = z + 10
    print(w.requires_grad)
    w.backward()
    print(x.grad)
    print(y.grad)