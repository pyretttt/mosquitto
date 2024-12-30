import math

def f(x, y):
    return math.sin(x * y) - x - y

def g(x, y):
    return math.exp(2 * x) - 2 * x + 3 * y


def findSolution(x0, y0, n=100):
    x_n, y_n = x0, y0
    for i in range(n):
        fx_n = fx(x_n, y_n)
        fy_n = fy(x_n, y_n)
        gx_n = gx(x_n, y_n)
        gy_n = gy(x_n, y_n)
        D_n = fx_n * gy_n - fy_n * gx_n
        x_n -= (f(x_n, y_n) * gy_n - g(x_n, y_n) * fy_n) / D_n
        y_n -= (f(x_n, y_n) * gx_n - g(x_n, y_n) * fx_n) / D_n
        print("New x_n: ", x_n, " y_n: ", y_n)
    
    return x_n, y_n

def fx(x, y):
    return y * math.cos(x * y) - 1


def fy(x, y):
    return x * math.cos(x * y) - 1

def gx(x, y):
    return 2 * math.exp(2 * x) - 2

def gy(x, y):
    return 3

if __name__ == "__main__":
    findSolution(0, 0)