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

# 飽和関数 #
def sat(x):
    return np.sign(x) * np.min([1, np.abs(x)])


if __name__ == '__main__':
    # サンプリング周期と時間ベクトル #
    dt = 0.01
    t = np.linspace(0, dt, 2000)

    V = 10.0
    def f(x):
        return np.array([[V*np.sin(x[1])], [0]]).reshape(2,1)

    B = np.array([[0], [1]])
    # func just for simulation
    def F(x, u):
        return np.array([[V*np.cos(x[2])], [V*np.sin(x[2])], [u]], dtype=np.float64)

    # 初期状態 #
    x = np.zeros([3, len(t)])
    x0 = 2 * np.random.random(2).reshape(2,1) - 0.5
    x0[0] *= 50
    x0[1] *= np.pi
    x[1:3,0] = x0.flatten()

    u = np.zeros(len(t))
    # 切換平面のパラメータ #
    z = V + 0.5
    def S(x):
        return x[0] + V*np.sin(x[1]) + z * np.remainder(x[1], 2*np.pi)

    # 符号関数 #
    alpha = 5
    def sig(S):
        return sat(alpha*S)

    # スライディングモード制御のゲイン
    K = 15.0

    # start simulation
    for k in range(len(t)-1):
        # ∂S/∂x #
        Sx = np.array([1, V*np.cos(x[2,k]) + z])

        try:
            u[k] = - np.dot(1/(np.dot(Sx,B)), (np.dot(Sx,f(x[1:3,k])) + [K*sig(S(x[1:3,k]))]))
        except Exception as e:
            from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
            raise(e)
        #u[k] = - np.linalg.inv(np.dot(p.T, B)) - np.dot(K, sig(S(xsmc[:,k].reshape(2,1))))
        x[:, k+1] = rk4(x[:,k].reshape(3,1), u[k], dt, F).flatten()



    #plt.clear()
    plt.subplot(4,1,1)
    plt.plot(t, x[0, :])
    plt.ylabel("state x1")
    plt.grid('on')
    plt.title("time series")
    plt.subplot(4,1,2)
    plt.plot(t, x[1,:])
    plt.ylabel('state x2')
    plt.grid('on')
    plt.subplot(4,1,3)
    plt.plot(t, u)
    plt.ylabel('input')
    plt.grid('on')

    plt.subplot(4,1,4)
    theta = np.pi*np.arange(-1,1,0.01);
    plt.plot(-V*np.sin(theta)-z*theta, theta, "k--", label = "sliding surface");
    #plt.plot(np.arange(-1,5), -p[0]/p[1]*np.arange(-1,5), "k--", label = "sliding surface");
    plt.plot(x[1,:], x[2,:], label = "state trajectory");
    plt.xlabel("state x1");
    plt.ylabel("state x2");
    plt.legend();
    plt.title("state space");
    plt.grid("on");
    plt.show()


