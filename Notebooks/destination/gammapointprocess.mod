COMMENT

Spike generator following a gamma process.

Gamma distributed random variables are generated using the Marsaglia-Tang method:

  G. Marsaglia and W. Tsang (2000) A simple method for generating gamma variables.
  ACM Transactions on Mathematical Software, 26(3):363-372. doi:10.1145/358407.358414

Parameters:
    alpha:     shape parameter of the gamma distribution. 1 = Poisson process.
    beta:      rate parameter of gamma distribution (/ms). Note that the mean interval is alpha/beta
    start:     start of gamma process (ms)
    duration:  length in ms of the spike train.

Author: Andrew P. Davison, UNIC, CNRS

ENDCOMMENT

NEURON  {
    ARTIFICIAL_CELL GammaProcess
    RANGE alpha, beta, start, duration
}

PARAMETER {
    alpha = 1                     : shape parameter of gamma distribution. 1 = Poisson process.
    beta = 0.1 (/ms) <1e-9,1e9>   : rate parameter of gamma distribution
                                  : mean interval is alpha/beta
    start = 1 (ms)                : start of first spike
    duration = 1000 (ms)          : input duration
}

ASSIGNED {
    event (ms)
    on
    end (ms)
}

PROCEDURE seed(x) {
    set_seed(x)
}

INITIAL {
    on = 0
    event = rand_gamma(alpha, beta)
    if (start >= 0) {
        net_send(event, 2)
    }
}

PROCEDURE event_time() {
    event = event + rand_gamma(alpha, beta)
    if (event > end) {
        on = 0
    }
}

NET_RECEIVE (w) {
    if (flag == 2) { : from INITIAL
        if (on == 0) {
            on = 1
            event = t
            end = t + 1e-6 + duration
            net_send(0, 1)
        }
    }
    if (flag == 1 && on == 1) {
        net_event(t)
        event = event + rand_gamma(alpha, beta)
        if (event > end) {
            on = 0
        }
        if (on == 1) {
            net_send(event - t, 1)
        }
    }
}

FUNCTION rand_gamma(alpha(1), beta(/ms)) (1) {
    LOCAL i, Z, U, T
    i = 1
    Z = 1.0
    while (i <= alpha){
        U = scop_random()
        Z = Z*U
        i = i + 1
    }
    T = -log(Z)/alpha
    i = 1
    while (T>0){
        T = T - beta*dt
        i = i + 1
    }

    rand_gamma = i*dt
}
