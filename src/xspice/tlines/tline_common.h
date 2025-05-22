/* tline_common.h
 * common definitions for all transmission lines
 * (c) Vadim Kuznetsov 2025
 */


#ifndef TLINE_COMMON_H
#define TLINE_COMMON_H

// Constants
#define Z0 (120*M_PI)
#define z0 50.0
#define MU0 (4*M_PI*1e-7)

#define C0 299792458.0

#define GMIN 1e-12


// Functions
#define sqr(x) (x*x)
#define cubic(x) (x*x*x)
#define quadr(x) (x*x*x*x)

#define coth(x) (1.0/tanh(x))
#define sech(x) (1.0/cosh(x))
#define cosech(x) (1.0/sinh(x))

// Data structures to hold transient state

typedef struct tline_state {
    double time;
    double I1;
    double I2;
    double V1;
    double V2;

    struct tline_state *next;
} tline_state_t;

// Functions to retrieve previous transient state
void append_state(tline_state_t **first, double time, double V1, double V2,
		double I1, double I2, double tmax);

tline_state_t *get_state(tline_state_t *first, double time);

#define PORT_NUM 4

typedef struct cpline_state {
    double time;
    double Ip[PORT_NUM];
    double Vp[PORT_NUM];

    struct cpline_state *next;
} cpline_state_t;

void append_cpline_state(cpline_state_t **first, double time, double *Vp, double *Ip, double tmax);

cpline_state_t *find_cpline_state(cpline_state_t *first, double time);



#endif
