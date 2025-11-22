import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(iter, m_init, b_init, learning_rate, x, y):
    m_curr = m_init
    b_curr = b_init
    n = len(x)
    for i in range(iter):
        y_predict = m_curr * x + b_curr
        error = (1/n)*sum([i**2 for i in (y - y_predict)])
        plt.plot([x[0], x[-1]], [m_curr * x[0] + b_curr, m_curr * x[-1] + b_curr])
        plt.pause(0.05)
        m_deriv = -(2/n)*sum(x*(y-y_predict))
        b_deriv = -(2/n)*sum(y-y_predict)
        m_curr = m_curr - learning_rate*m_deriv
        b_curr = b_curr - learning_rate*b_deriv
        print(f"iteration: {i}, m: {m_curr}, b:{b_curr}, error: {error}" )
        if error < 1e-10:
            print("Algorithm converged! Ending the loop!")
            break
    plt.show()

def plot_GD():
    pass


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7,9, 11, 13])

gradient_descent(100, -1, -8, 0.01, x, y)