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
from libs.cython.cython_equations import c_eq_I_V_lambertW
from timeit import default_timer as timer


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

    rnd_uniq = 2*np.random.random(size=Volts.size) - 1
    y_noise = 0.0* Isc_inp * rnd_uniq

    N = np.size(Volts)
    Ical = np.zeros(N)
    i = 0
    for V in Volts:
        # arg = q*Rs/(n*kB*T)*(Isc - Voc/(Rs+Rsh))*np.exp(-q*Voc/(n*kB*T))* np.exp(q*(Rs*Isc+Rsh*V/(Rs+Rsh))/(n*kB*T))
        # Ical[i] = n*kB*T/(q*Rs)*lambertw(arg) + V/Rs - Isc - Rsh*V/(Rs*(Rs+Rsh))

        # =============================
        # =============================

        # arg1 = Rsh*( Rs*( (Isc+(Rs*Isc - Voc)/Rsh)/(1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T))) ) + Rs*Voc/Rsh + V )
        #
        # arg2 = arg1/ (Rs*(Rs+Rsh))
        #
        # argLambert1 = (q*Rs/(n*kB*T)*(Isc - Voc/(Rs+Rsh))*np.exp(-q*Voc/(n*kB*T)) ) / \
        #              (1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T)))
        #
        # argLambert2 = np.exp( Rsh*q*( Rs*( (Isc+(Rs*Isc - Voc)/Rsh)/(1 - np.exp(q*(Rs*Isc - Voc)/(n*kB*T))) ) + Rs*Voc/Rsh + V ) /
        #                       (n*kB*T*(Rs + Rsh)) )
        #
        # LambertFunc = lambertw(argLambert1*argLambert2)
        #
        # arg3 = n*kB*T/q/Rs*LambertFunc
        #
        # Ical[i] = V/Rs -  arg2 + arg3
        #
        # =============================

        Ical[i] = -(
            -q * V + (-(lambertw(
            q * b[1] * (Isc - (Voc - b[1] * Isc) / b[2]) * np.exp(-q * Voc / (b[0] * kB * T)) * b[2] /
            (b[1] * b[0] * kB * T + b[2] * b[0] * kB * T) * np.exp(b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V)
                                                                / b[0] / kB / T / (b[2] + b[1])))) +
            b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V) / b[0] / kB / T / (b[2] + b[1])) * b[0] * kB * T
                    ) / q / b[1]


        i = i+1
    return -Ical + y_noise

def func(b, I, V, Voc_inp, Isc_inp):
    return I - fun_for_minimization(b, V, Voc_inp, Isc_inp)



def func_abs(b, I, V, Voc_inp, Isc_inp):
    return np.sum(np.abs(fun_for_minimization(b, V, Voc_inp, Isc_inp) - I))



def func_abs_c(b, I, V, Voc_inp, Isc_inp):
    # with cython implementation:
    return 1e4*np.sum(np.abs( np.real(c_eq_I_V_lambertW(b, V, Voc_inp, Isc_inp, 0.01)) - I ))



if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')

    data = np.loadtxt('data/reference_Si.dat')
    x = data[:, 0]
    y = data[:, 1]*100/1e4
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
    args_cf = (Voc, Isc)

    ax.plot(x0, 0.0, 'o', color = 'r', label='Voc={:1.3e}'.format(Voc))
    ax.plot(0.0, y0, 'o', color = 'b', label='Isc={:1.3e}'.format(Isc))

    ax.set_xlabel('Volts [V]')
    ax.set_ylabel('Current [A]')

    fig.tight_layout()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # ==================================


    b0 = np.asarray([1.65, 1.2e4, 0.6833e7], dtype=float) # for refe Si sample
    # b0 = np.asarray([1.5, 1.0e4, 0.684e6], dtype=float)
    # bounds = [(1, 2), (0, np.inf), (0, np.inf)]
    bounds = ([1, 1e2, 1e2], [2, 1e8, 1e8])
    # res = least_squares(func, b0, bounds=bounds, args=args, gtol=1e-12)

    def func_cf(I, a, b, c, Voc_inp=Voc, Isc_inp=Isc):
        # for curve_fit procedure:
        p0 = np.array([a, b, c])
        out = np.sum(np.abs(I - fun_for_minimization(p0, y, Voc_inp, Isc_inp)))
        return out

    popt, pcov = curve_fit(func_cf, xdata=x, ydata=y, p0=b0, bounds=bounds, absolute_sigma=True)
    print(popt)
    print(pcov)
    b = popt.data
    print('===' * 15)
    print('least_sq:' )
    print('n = {0:1.6f}, Rs0 = {1:1.6f}, Rsh0 = {2:1.6f}, std = {std:1.5e}'.format(b[0], b[1], b[2], std=np.sum(func_cf(y, b[0], b[1], b[2], Voc_inp=Voc, Isc_inp=Isc))))
    ax.plot(x, (fun_for_minimization(b, Volts=x, Voc_inp=Voc, Isc_inp=Isc)), '-g', label='CF')
    plt.legend()



    # ==================================
    #
    bounds = [(1, 2), (1e3, 1e8), (1e3, 1e8)]
    # bounds = ([1, 0.1, 0.1], [2, np.inf, np.inf])

    start = timer()
    result_c = differential_evolution(func_abs_c, bounds=bounds, args=args, disp=False, tol=1e-11, maxiter=int(1e2),
                                    strategy='randtobest1exp', )
    de_func_abs_c_time = timer() - start
    print("DE searching procedure of minimizing the func_abs_c tooks: {0:f} seconds".format(de_func_abs_c_time))

    start = timer()
    result = differential_evolution(func_abs, bounds=bounds, args=args, disp=False, tol=1e-11, maxiter=int(1e2), strategy='randtobest1exp',)
    de_func_abs_time = timer() - start
    print("DE searching procedure of minimizing the func_abs tooks: {0:f} seconds".format(de_func_abs_time))



    b_de = result.x
    b_c = result_c.x
    print('===' * 15)
    print('de:' * 15)
    print('n = {0:1.6f}, Rs0 = {1:1.6f}, Rsh0 = {2:1.6f}'.format(b_de[0], b_de[1], b_de[2]))
    print('c_de:' * 15)
    print('n = {0:1.6f}, Rs0 = {1:1.6f}, Rsh0 = {2:1.6f}'.format(b_c[0], b_c[1], b_c[2]))

    # b = np.asarray([1.5, 10000, 1000], dtype=float)
    ax.plot(x, (fun_for_minimization(b_de, Volts=x, Voc_inp=Voc, Isc_inp=Isc)), '-r', label='DE')
    ax.plot(x, c_eq_I_V_lambertW(b_c, x, Voc, Isc, 0.01), '-y', label='DE cython')
    ax.plot(x, c_eq_I_V_lambertW(np.array(b), x, Voc, Isc, 0.01), '-y', label='DE cython ideal')

    # ==================================




    plt.legend()
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    print('---'*15)
    print(fun_for_minimization(b=[1.5, 1000, 10000], Volts=np.array([0.03]), Voc_inp=0.5323, Isc_inp=-6.711e-5))
    print('---'*15)
    print(c_eq_I_V_lambertW(np.array([1.5, 1000, 10000]), np.array([0.03]), 0.5323, -6.711e-5, 0.0))

    print('***'*20)
    print("std CF function: {}".format(1e4*np.sum(func_cf(y, b[0], b[1], b[2], Voc_inp=Voc, Isc_inp=Isc))))
    print("std DE_c function: {}".format(func_abs_c(b_c, y, x, Voc_inp=Voc, Isc_inp=Isc)))

    plt.show()

    print('finish')