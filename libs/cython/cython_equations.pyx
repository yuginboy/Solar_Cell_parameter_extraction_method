import numpy
cimport numpy
cpdef numpy.ndarray[numpy.double_t, ndim=1] cVectorAdd_f(numpy.ndarray[numpy.double_t, ndim=1] a, numpy.ndarray[numpy.double_t, ndim=1] b):
    cdef int i, n
    n = numpy.size(a)
    cdef numpy.ndarray[numpy.double_t, ndim=1] out
    out = numpy.ndarray(n, dtype=numpy.double)

    for i in range(n):
        out[i] = a[i] + b[i]

    return out

# def cVectorAdd_f (a, b):
#     return a + b

cpdef numpy.ndarray[numpy.double_t, ndim=1] c_eq_I_V_lambertW(
        numpy.ndarray[numpy.double_t, ndim=1] b,
        numpy.ndarray[numpy.double_t, ndim=1] Volts,
        double Voc_inp,
        double Isc_inp,
        double rnd_scale,
                                                            ):
    cdef double   Voc, Isc, T, kB, q, n, Rs, Rsh
    cdef int N, i
    cdef numpy.ndarray[numpy.double_t, ndim=1] y_noise, Ical



    Voc = Voc_inp  # [V]
    Isc = Isc_inp  # [A]
    T = 300  # [K]
    kB = 1.38e-23  # [J/K]
    q = 1.6e-19  # [C]

    n   = b[0]
    Rs  = b[1]
    Rsh = b[2]

    N = numpy.size(Volts)
    y_noise = numpy.ndarray(N, dtype=numpy.double)
    Ical    = numpy.ndarray(N, dtype=numpy.double)

    y_noise = rnd_scale* Isc * 2*numpy.random.random(size=N) - 1

    Ical = numpy.zeros(N)
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
            -q * V + (-lambertw(
            q * b[1] * (Isc - (Voc - b[1] * Isc) / b[2]) * np.exp(-q * Voc / (b[0] * kB * T)) * b[2] /
            (b[1] * b[0] * kB * T + b[2] * b[0] * kB * T) * np.exp(b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V)
                                                                / b[0] / kB / T / (b[2] + b[1]))) +
            b[2] * q * (b[1] * (Isc + b[1] * Isc / b[2]) + V) / b[0] / kB / T / (b[2] + b[1])) * b[0] * kB * T
                    ) / q / b[1]


        i = i+1
    return -Ical + y_noise

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')