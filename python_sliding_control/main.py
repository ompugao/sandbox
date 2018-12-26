#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 4次Runge-Kutta法による状態遷移シミュレーション #
# 入力は零次ホールドを仮定 #
def rk4(x, u, dt, f):
    k1 = f(x, u)
    k2 = f(x+k1*dt/2, u)
    k3 = f(x+k2*dt/2, u)
    k4 = f(x+k3*dt, u)

    return x + dt/6 * (k1+2*k2+2*k3+k4)

if __name__ == '__main__':
    # サンプリング周期と時間ベクトル #
    dt = 0.01
    t = np.linspace(0, dt, 10)

    a = 1.0, b = 2.0
    def f(x):
        return np.array([[x[1], a*np.sin(x[0])]])

    B = np.array([[0, b]])
    def F(x, u):
        return f(x) + np.dot(B, u)

    # 初期状態 #
    x0 = np.array([[ 5.0, 0.0]])

    # 切換平面のパラメータ #
    p = np.array([[1.0, 2.0 ]])
    # 切換平面 #
    def S(x):
        return np.dot(p, x)

    # 符号関数 #
    alpha = 10
    def sig(S):
        return np.tanh(np.dot(alpha, S))
    # スライディングモード制御のゲイン #
    K = 2.0

    xsmc = np.zeros([2, len(t)])
    xsmc[:, 0] = x0
    usmc = np.zeros([1, len(t)])

    # start simulation
    for k in range(len(t)):
        usmc[k] = - np.linalg.inv(np.dot(p.T, B)) - np.dot(K, 

    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

    
    plt.show()
