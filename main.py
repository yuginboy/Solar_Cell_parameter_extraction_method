'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-06-09
'''
import numpy as np
from scipy.special import lambertw
from scipy.optimize import least_squares, curve_fit, differential_evolution
import matplotlib.pylab as plt

# V = 1 #[V]
# I = 0.1 #[A]


def fun_for_minimization(b, Volts, Voc_inp, Isc_inp):
    Voc = Voc_inp  # [V]
    Isc = Isc_inp  # [A]
    T = 300  # [K]
    kB = 1.38e-23  # [J/K]
    q = 1.6e-19  # [C]

    n = b[0]
    Rs = b[1]
    Rsh = b[2]

    N = np.size(Volts)
    Ical = np.zeros(N)
    i = 0
    for V in Volts:
        # arg = q*Rs/(n*kB*T)*(Isc - Voc/(Rs+Rsh))*np.exp(-q*Voc/(n*kB*T))* np.exp(q*(Rs*Isc+Rsh*V/(Rs+Rsh))/(n*kB*T))
        # Ical[i] = n*kB*T/(q*Rs)*lambertw(arg) + V/Rs - Isc - Rsh*V/(Rs*(Rs+Rsh))

        arg1 = Rsh*( Rs*( (Isc+(Rs*Isc - Voc)/Rsh)/(1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T))) ) + Rs*Voc/Rsh + V )

        arg2 = arg1/ (Rs*(Rs+Rsh))

        argLambert1 = (q*Rs/(n*kB*T)*(Isc - Voc/(Rs+Rsh))*np.exp(-q*Voc/(n*kB*T)) ) / \
                     (1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T)))

        argLambert2 = np.exp( Rsh*q*( Rs*( (Isc+(Rs*Isc - Voc)/Rsh)/(1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T))) ) + Rs*Voc/Rsh + V ) /
                              (n*kB*T*(Rs + Rsh)) )

        LambertFunc = lambertw(argLambert1*argLambert2)

        arg3 = n*kB*T/q/Rs*LambertFunc

        Ical[i] = 0*V/Rs -  arg2 + 0*arg3

        # Ical[i] = -(
        #     -q * V + (-lambertw(
        #     q * b[1] * (Isc - (Voc - b[1] * Isc) / b[2]) * np.exp(-q * Voc / (b[0] * kB * T)) * b[2] /
        #     (b[1] * b[0] * kB * T + b[2] * b[0] * kB * T) * np.exp(b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V)
        #                                                         / b[0] / kB / T / (b[2] + b[1]))) +
        #     b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V) / b[0] / kB / T / (b[2] + b[1])) * b[0] * kB * T
        #             ) / q / b[1]
        i = i+1
    return Ical

def func(b, I, V, Voc_inp, Isc_inp):
    return I - fun_for_minimization(b, V, Voc_inp, Isc_inp)

def func_abs(b, I, V, Voc_inp, Isc_inp):
    return np.sum(np.abs(fun_for_minimization(b, V, Voc_inp, Isc_inp) - I))



if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')


    data = np.loadtxt('/home/yugin/VirtualboxShare/Giorgi.Tchutchulashvili/src/data/Dark_and_Ilum_AUPD1.dat')
    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, label='experiment')
    ax = plt.gca()
    ax.grid(True, which='both')

    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')


    maxX = np.max(x)
    minX = np.min(x)

    maxY = np.max(y)
    minY = np.min(y)

    y0 = np.interp(0.0, x, y)
    x0 = np.interp(0.0, y, x)

    Voc = x0
    Isc = y0
    args = (x, y, Voc, Isc)

    ax.plot(x0, 0.0, 'o', color = 'r', label='Voc={:1.3e}'.format(Voc))
    ax.plot(0.0, y0, 'o', color = 'b', label='Isc={:1.3e}'.format(Isc))

    ax.set_xlabel('Volts [V]')
    ax.set_ylabel('Current [A]')

    fig.tight_layout()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()


    # b0 = np.asarray([1.7, 1000, 1000], dtype=float)
    # # bounds = [(1, 2), (0, np.inf), (0, np.inf)]
    # bounds = ([1, 0.1, 0.1], [2, 1e12, 1e12])
    # res = least_squares(func, b0, bounds=bounds, args=args, gtol=1e-12)
    # print(res)
    # b = res.x
    # print('===' * 15)
    # print('least_sq:' * 15)
    # print('n = {0:1.6f}, Rs0 = {1:1.6f}, Rsh0 = {2:1.6f}'.format(b[0], b[1], b[2]))
    # ax.plot(x, (fun_for_minimization(res.x, Volts=x, Voc_inp=Voc, Isc_inp=Isc)), '-g', label='LS')
    # plt.legend()

    bounds = [(1, 2), (0, 1e5), (0, 1e6)]
    # bounds = ([1, 0.1, 0.1], [2, np.inf, np.inf])

    result = differential_evolution(func_abs, bounds=bounds, args=args, disp=True, tol=1e-8)

    b = result.x
    print('===' * 15)
    print('de:' * 15)
    print('n = {0:1.6f}, Rs0 = {1:1.6f}, Rsh0 = {2:1.6f}'.format(b[0], b[1], b[2]))

    # b = np.asarray([1.5, 10000, 1000], dtype=float)
    ax.plot(x, (fun_for_minimization(b, Volts=x, Voc_inp=Voc, Isc_inp=Isc)), '-r', label='DE')
    plt.legend()
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    print(fun_for_minimization(b=[1.5, 1000, 10000], Volts=np.array([0.03]), Voc_inp=0.5323, Isc_inp=-6.711e-5))

    plt.show()

    print('finish')