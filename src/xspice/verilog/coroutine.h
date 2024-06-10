#ifndef _COSIM_COROUTINE_H_
#define _COSIM_COROUTINE_H_

#if defined(__MINGW32__) || defined(_MSC_VER)
#include <windows.h> // Windows has a simple co-routine library - "Fibers"

/* Coroutine context information. */

struct cr_ctx {
    LPVOID              spice_fiber;    // OS-provided coroutine context.
    LPVOID              cosim_fiber;    // OS-provided context and stack.
};
#else
/* On a Unix-like OS pthreads are used to give a co-simulation its own stack,
 * but setcontext() and friends would avoid the overhead of a OS thread.
 */

#include <pthread.h>

struct cr_ctx {
    pthread_t           thread;         // Thread for VVP execution.
    pthread_mutex_t     mutex;
    pthread_cond_t      spice_cond;     // Condition variables for each thread.
    pthread_cond_t      cosim_cond;
};

#endif /* pthread code. */
#endif // _COSIM_COROUTINE_H_

