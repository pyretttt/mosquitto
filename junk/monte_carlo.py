import math
import numpy as np

def monteCarlo(n, fn, region):
    x1, x2, y1, y2 = region
    
    x_points = np.linspace(x1, x2, n)
    y_points = np.linspace(y1, y2, n)
    
    overline_f = 0.0
    overline_f2 = 0.0
    hyper_volume = (x2 - x1) * (y2 - y1)
    for (x, y) in zip(x_points, y_points):
        out = fn(x, y)
        overline_f += out / n
        overline_f2 += out ** 2 / n
        
    std = (hyper_volume / n) * np.sqrt((overline_f2 - overline_f) / n)
    
    return hyper_volume * overline_f, std


if __name__ == '__main__':
    def fn(x, y):
        return np.exp(x * y)
    
    for n in [10, 100, 1000, 10000, 100000, 1000000]:
        exp, pm = monteCarlo(n, fn, (0, 1, 0, 1))
        
        print(f"N = {n}, integral = {exp} +/- {pm}")