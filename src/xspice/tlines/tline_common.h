/* tline_common.h
 * common definitions for all transmission lines
 */

/* ===========================================================================
 FILE   tline_common.h
 Copyright 2025 Vadim Kuznetsov

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#ifndef TLINE_COMMON_H
#define TLINE_COMMON_H

// Constants

#ifdef _MSC_VER
#ifndef M_E
  #define M_E        2.7182818284590452354    // e
#endif
#ifndef M_PI
  #define M_PI       3.14159265358979323846   // pi
#endif
#define M_LN2      0.69314718055994530942   // log_e 2
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#endif

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

void delete_tline_states(tline_state_t **first);
void delete_tline_last_state(tline_state_t **first);
tline_state_t *get_tline_last_state(tline_state_t *first);

#define PORT_NUM 4

typedef struct cpline_state {
    double time;
    double Ip[PORT_NUM];
    double Vp[PORT_NUM];

    struct cpline_state *next;
} cpline_state_t;

void append_cpline_state(cpline_state_t **first, double time, double *Vp, double *Ip, double tmax);

cpline_state_t *find_cpline_state(cpline_state_t *first, double time);


void delete_cpline_states(cpline_state_t **first);
void delete_cpline_last_state(cpline_state_t **first);
cpline_state_t *get_cpline_last_state(cpline_state_t *first);

#endif
