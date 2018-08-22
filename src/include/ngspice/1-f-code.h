#include "ngspice/randnumb.h"

void f_alpha(int n_pts, int n_exp, double X[], double Q_d,
double alpha);

void rvfft(float X[], unsigned long int n);


#define TRNOISE_STATE_MEM_LEN 4
struct trnoise_state
{
    double points[TRNOISE_STATE_MEM_LEN];
    size_t top;

    double NA, TS, NAMP, NALPHA, RTSAM, RTSCAPT, RTSEMT;

    double *oneof;
    size_t oneof_length;

    bool timezero;

    double RTScapTime, RTSemTime;
    bool RTS;
};

struct trrandom_state
{
    double value;

    int rndtype;
    double TS, TD, PARAM1, PARAM2;
};

struct trnoise_state *trnoise_state_init(double NA, double TS, double NALPHA, double NAMP, double RTSAM, double RTSCAPT, double RTSEMT);
struct trrandom_state *trrandom_state_init(int rndtype, double TS, double TD, double PARAM1, double PARAM2);


void trnoise_state_gen(struct trnoise_state *this, CKTcircuit *ckt);
void trnoise_state_free(struct trnoise_state *this);


static inline void
trnoise_state_push(struct trnoise_state *this, double val)
{
    this->points[this->top++ % TRNOISE_STATE_MEM_LEN] = val;
}


static inline double
trnoise_state_get(struct trnoise_state *this, CKTcircuit *ckt, size_t index)
{
    while(index >= this->top)
        trnoise_state_gen(this, ckt);

    if(index + TRNOISE_STATE_MEM_LEN < this->top) {
        fprintf(stderr, "ouch, trying to fetch from the past %d %d\n",
                (int)index, (int)this->top);
        controlled_exit(1);
    }

    return this->points[index % TRNOISE_STATE_MEM_LEN];
}

static inline double
trrandom_state_get(struct trrandom_state *this)
{
    double param1 = this->PARAM1;
    double param2 = this->PARAM2;
    switch (this->rndtype) {
        case 1:
            /* param1: range -param1[ ... +param1[  (default = 1)
               param2: offset  (default = 0)
            */
            return (param1 * drand() + param2);
            break;
        case 2:
            /* param1: standard deviation (default = 1)
               param2: mean  (default = 0)
            */
            return param1 * gauss0() + param2;
            break;
        case 3:
            /* param1: mean (default = 1)
               param2: offset  (default = 0)
            */
            return exprand(param1) + param2;
            break;
        case 4:
            /* param1: lambda (default = 1)
               param2: offset  (default = 0)
            */
            return (double)poisson(param1) + param2;
            break;

        default:
            return 0.0;
            break;
    }
}

