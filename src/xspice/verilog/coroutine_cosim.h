#ifndef _COSIM_COROUTINE_SIM_H_
#define _COSIM_COROUTINE_SIM_H_

/* Code that supplies a set of simple, portable functions for running
 * a co-simulation as a co-routine inside Ngspice: co-simulator side.
 */

#include "coroutine.h"

#if defined(__MINGW32__) || defined(_MSC_VER)
static void cr_yield_to_spice(struct cr_ctx *ctx) {
    SwitchToFiber(ctx->spice_fiber);
}

#define cr_init(X) /* All initialisation was done in the primary fiber. */
#else
/* On a Unix-like OS pthreads are used to give libvvp its own stack. */

static void cr_yield_to_spice(struct cr_ctx *ctx) {
    pthread_cond_signal(&ctx->spice_cond);
    pthread_cond_wait(&ctx->cosim_cond, &ctx->mutex);
}

static void cr_init(struct cr_ctx *ctx) {
    pthread_mutex_lock(&ctx->mutex);
}
#endif
#endif // _COSIM_COROUTINE_SIM_H_
