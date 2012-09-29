/* Copyright: Holger Vogt, 2008
 Generates 1/f noise values according to:
 "Discrete simulation of colored noise and stochastic
 processes and 1/fa power law noise generation"
 Kasdin, N.J.;
 Proceedings of the IEEE
 Volume 83,  Issue 5,  May 1995 Page(s):802 - 827
*/

#include "ngspice/ngspice.h"
#include "ngspice/cpextern.h"
#include "ngspice/cktdefs.h"
#include "ngspice/1-f-code.h"

#include "ngspice/fftext.h"
#include "ngspice/wallace.h"


extern void controlled_exit(int status);


void
f_alpha(int n_pts, int n_exp, double X[], double Q_d, double alpha)
{
    int i;
    double *hfa, *wfa;
    double ha;

    ha = alpha/2.0f;
    // Q_d = sqrt(Q_d); /* find the deviation of the noise */
    hfa = TMALLOC(double,n_pts);
    wfa = TMALLOC(double,n_pts);
    hfa[0] = 1.0f;
    wfa[0] = Q_d * GaussWa;
    /* generate the coefficients hk */
    for (i = 1; i < n_pts; i++) {
        /* generate the coefficients hk */
        hfa[i] = hfa[i-1] * (ha + (double)(i-1)) / ((double)(i));
        /* fill the sequence wk with white noise */
        wfa[i] = Q_d * GaussWa;
    }

    // for (i = 0; i < n_pts; i++)
    //    printf("rnd %e, hk %e\n", wfa[i], hfa[i]);

    /* perform the discrete Fourier transform */
    fftInit(n_exp);
    rffts(hfa, n_exp, 1);
    rffts(wfa, n_exp, 1);

    /* multiply the two complex vectors */
    rspectprod(hfa, wfa, X, n_pts);
    /* inverse transform */
    riffts(X, n_exp, 1);

    free(hfa);
    free(wfa);
    /* fft tables will be freed in vsrcaccept.c and isrcaccept.c
       fftFree(); */
    fprintf(stdout, "%d (2e%d) one over f values created\n", n_pts, n_exp);
}


/*-----------------------------------------------------------------------------*/

void
trnoise_state_gen(struct trnoise_state *this, CKTcircuit *ckt)
{
    if (this->top == 0) {

        if (cp_getvar("notrnoise", CP_BOOL, NULL))
            this -> NA = this -> TS = this -> NALPHA = this -> NAMP =
                this -> RTSAM = this -> RTSCAPT = this -> RTSEMT = 0.0;

        if ((this->NALPHA > 0.0) && (this->NAMP > 0.0)) {

            // add 10 steps for start up sequence
            size_t nosteps = (size_t) (ckt->CKTfinalTime / this->TS) + 10;

            size_t newsteps = 1;
            int newexp = 0;

            // generate number of steps as power of 2
            while (newsteps < nosteps) {
                newsteps <<= 1;
                newexp++;
            }

            this->oneof = TMALLOC(double, newsteps);
            this->oneof_length = newsteps;

            f_alpha((int) newsteps, newexp,
                    this -> oneof,
                    this -> NAMP,
                    this -> NALPHA);
        }

        trnoise_state_push(this, 0.0); /* first is deterministic */
        return;
    }


    // make use of two random variables per call to rgauss()
    {
        double ra1, ra2;
        double NA = this -> NA;

        if (NA != 0.0) {

#ifdef FastRand
            // use FastNorm3
            ra1 = NA * FastNorm;
            ra2 = NA * FastNorm;
#elif defined(WaGauss)
            // use WallaceHV
            ra1 = NA * GaussWa;
            ra2 = NA * GaussWa;
#else
            rgauss(&ra1, &ra2);
            ra1 *= NA;
            ra2 *= NA;
#endif

        } else {

            ra1 = 0.0;
            ra2 = 0.0;

        }

        if (this -> oneof) {

            if (this->top + 1 >= this->oneof_length) {
                fprintf(stderr, "ouch, noise data exhausted\n");
                controlled_exit(1);
            }

            ra1 += this->oneof[this->top]      -  this->oneof[0];
            ra2 += this->oneof[this->top + 1]  -  this->oneof[0];
        }

        trnoise_state_push(this, ra1);
        trnoise_state_push(this, ra2);
    }
}


struct trnoise_state *
trnoise_state_init(double NA, double TS, double NALPHA, double NAMP, double RTSAM, double RTSCAPT, double RTSEMT)
{
    struct trnoise_state *this = TMALLOC(struct trnoise_state, 1);

    this->NA = NA;
    this->TS = TS;
    this->NALPHA = NALPHA;
    this->NAMP = NAMP;
    this->RTSAM = RTSAM;
    this->RTSCAPT = RTSCAPT;
    this->RTSEMT = RTSEMT;

    if (RTSAM > 0) {
        this->RTScapTime = exprand(RTSCAPT);
        this->RTSemTime = this->RTScapTime + exprand(RTSEMT);
    }

    this -> top = 0;
    this -> oneof = NULL;

    return this;
}


struct trrandom_state *
trrandom_state_init(int rndtype, double TS, double TD, double PARAM1, double PARAM2)
{
    struct trrandom_state *this = TMALLOC(struct trrandom_state, 1);

    this->rndtype = rndtype;
    this->TS = TS;
    this->TD = TD;
    this->PARAM1 = PARAM1;
    this->PARAM2 = PARAM2;
    this->value = 0.0;

    return this;
}
