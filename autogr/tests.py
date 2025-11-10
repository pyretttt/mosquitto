from tensor import Tensor
import numpy as np

if __name__ == "__main__":
    x = Tensor.from_numpy(np.array([1, 2, 3]).reshape(3, 1), requires_grad=True)
    y = x + 5
    z = y + 10
    print(z.requires_grad)
    z.backward()
    print(x.grad)