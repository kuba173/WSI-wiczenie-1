import numpy as np
import matplotlib.pyplot as plt
import random
import timeit

from numpy.linalg import inv

def booth(x,y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def booth_dx(x,y):
    return 2*(x+2*y-7)+2*2*(2*x+y-5)

def booth_dy(x,y):
    return 4*(x+2*y-7)+2*(2*x+y-5)

def booth_dxdx(x,y):
    return 2+2*2*2

def booth_dydy(x,y):
    return 4*2+2

def booth_dxdy(x,y):
    return 2*2+2*2

def booth_hessian(x, y):
    return np.array([[booth_dxdx(x, y), booth_dxdy(x, y)], [booth_dxdy(x, y), booth_dydy(x, y)]])


def gradient(booth_dx,booth_dy,x0,y0,B,n_iter,eps):

    x=x0
    y=y0

    prev_x = x0
    prev_y = y0
    for i in range(n_iter):

        x=x-B*booth_dx(x,y)
        y=y-B*booth_dy(x,y)

        if abs(x - prev_x) < eps and abs(y - prev_y) < eps:
            print("Iteration gradient e:", i)
            break

        prev_x = x
        prev_y = y

    return x, y

############
def remove_duplicates(vector):
    return list(set(vector))

def newton(booth_dx, booth_dy, booth_hessian, x0, y0,B, n_iter, eps):

    x = x0
    y = y0

    prev_x = x0
    prev_y = y0
    for i in range(n_iter):
        dx, dy = booth_dx(x,y), booth_dy(x,y)
        hess = booth_hessian(x,y)

        grad = np.array([dx, dy], dtype=float)

        hess_inv = inv(hess)

        delta = -hess_inv.dot(grad)

        x_new = np.array([x, y]) + B*delta

        x,y = x_new

        if abs(x - prev_x) < eps and abs(y - prev_y) < eps:
            print("Iteration newton e:", i)
            break

        prev_x = x
        prev_y = y
    return x,y

############



ro=2
n_iter=1000
B=0.1
tolerance = 0.1
eps=1e-6
similar_x = 0
similar_y = 0

results_x = []
results_y = []

results_x_n = []
results_y_n = []

x_l = np.linspace(-5, 5, 100)
y_l = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_l, y_l)
Z = booth(X, Y)

for i in range(10):

    x0 = random.uniform(-5, 5)
    y0 = random.uniform(-5, 5)

    print("======================================")
    print("test nr",i+1)
    print("Initial value")
    print("x0 = ",x0)
    print("y0 = ",y0)
    start_time = timeit.default_timer()
    x, y = gradient(booth_dx, booth_dy, x0, y0, B, n_iter, eps)
    end_time = timeit.default_timer()
    time_g = end_time - start_time
    x=round(x, ro)
    y=round(y, ro)

    results_x.append(x)
    results_y.append(y)
    points = [(x, y) for x, y in zip(results_x, results_y)]
    unique_points_gradient = set(points)

    print("result for gradient")
    print("x =", x)
    print("y =", y)
    print("time =", time_g)

    start_time = timeit.default_timer()
    x, y = newton(booth_dx, booth_dy, booth_hessian, x0, y0, B, n_iter, eps)
    end_time = timeit.default_timer()
    time_n = end_time - start_time
    x = round(x, ro)
    y = round(y, ro)

    results_x_n.append(x)
    results_y_n.append(y)
    points = [(x, y) for x, y in zip(results_x_n, results_y_n)]
    unique_points_newton = set(points)

    print("result for newton")
    print("x =", x)
    print("y =", y)
    print("time =", time_n)
    print("======================================")


print("|||||||||||||||||||||||||||||||||||||||||||||")
print("conclusion")
print("|||||||||||||||||||||||||||||||||||||||||||||")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

print("minimum for gradient")
for p_x,p_y in unique_points_gradient:

    print("f(" ,p_x," , ",p_y,") = ",round(booth(p_x, p_y),2))
    ax.scatter(results_x, results_y, s=50, c='red')

print("minimum for newton")
for p_x,p_y in unique_points_newton:

    print("f(" ,p_x," , ",p_y,") = ",round(booth(p_x, p_y),2))
    ax.scatter(results_x_n, results_y_n, s=50, c='blue')
# plot


plt.show()
