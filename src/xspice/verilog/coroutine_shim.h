#ifndef _COSIM_COROUTINE_SHIM_H_
#define _COSIM_COROUTINE_SHIM_H_

/* Code that supplies a set of simple, portable functions for running
 * a co-simulation as a co-routine inside Ngspice: d_cosim shim side.
 * For Windows it also emulates the Unix dlopen() family of functions
 * for dynamic loading.
 */

#include "coroutine.h"

static void fail(const char *what, int why);

#if defined(__MINGW32__) || defined(_MSC_VER)
#define dlopen(name, type) LoadLibrary(name)
#define dlsym(handle, name) (void *)GetProcAddress(handle, name)
#define dlclose(handle) FreeLibrary(handle)
#define cr_safety() while (0)     // Not needed with Fibers.

static char *dlerror(void) // Lifted from dev.c.
{
    static const char errstr_fmt[] =
         "Unable to find message in dlerr(). System code = %lu";
    static char errstr[256];
    LPVOID lpMsgBuf;

    DWORD rc = FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            GetLastError(),
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR) &lpMsgBuf,
            0,
            NULL
            );

    if (rc == 0) { /* FormatMessage failed */
        (void) sprintf(errstr, errstr_fmt, (unsigned long) GetLastError());
    } else {
        snprintf(errstr, sizeof errstr, "%s", (char *)lpMsgBuf);
        LocalFree(lpMsgBuf);
    }
    return errstr;
} /* end of function dlerror */

static void cr_yield_to_sim(struct cr_ctx *ctx) {
    SwitchToFiber(ctx->cosim_fiber);
}

static void cr_init(struct cr_ctx *ctx, void *(*fn)(void *), void *data) {
    ctx->spice_fiber = ConvertThreadToFiber(NULL);

    /* Start the cosimulator fiber and wait for it to be ready. */

    ctx->cosim_fiber = CreateFiber(1024*1024, (void (*)(void *))fn, data);
    cr_yield_to_sim(ctx);
}

static void cr_cleanup(struct cr_ctx *ctx) {
    DeleteFiber(ctx->cosim_fiber);
}

static void cr_yield_to_spice(struct cr_ctx *ctx) {
    SwitchToFiber(ctx->spice_fiber);
}

#else // Pthreads

#include <dlfcn.h>
#include <signal.h>

static void cr_yield_to_sim(struct cr_ctx *ctx) {
    int err;

    err = pthread_cond_signal(&ctx->cosim_cond);
    if (err)
        fail("pthread_cond_signal (sim)", err);
    err = pthread_cond_wait(&ctx->spice_cond, &ctx->mutex);
    if (err)
        fail("pthread_cond_wait (spice)", err);
}

static void cr_init(struct cr_ctx *ctx, void *(*fn)(void *), void *data) {
    int err;

    /* Create pthread apparatus. */

    err = pthread_mutex_init(&ctx->mutex, NULL);
    if (err)
        fail("pthread_mutex_init", err);
    err = pthread_cond_init(&ctx->spice_cond, NULL);
    if (!err)
        err = pthread_cond_init(&ctx->cosim_cond, NULL);
    if (err)
        fail("pthread_cond_init", err);

    /* Start the cosimulator thread and wait for it to be ready. */

    pthread_mutex_lock(&ctx->mutex);
    err = pthread_create(&ctx->thread, NULL, fn, data);
    if (err)
        fail("pthread_create", err);
    err = pthread_cond_wait(&ctx->spice_cond, &ctx->mutex);
    if (err)
        fail("pthread_cond_wait", err);
}

static void cr_safety(void) {
    sigset_t set;

    /* Signals that are handled with longjump() in signal_handler.c
     * must be blocked to prevent threads sharing a stack.
     */

    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGFPE);
    sigaddset(&set, SIGTTIN);
    sigaddset(&set, SIGTTOU);
    sigaddset(&set, SIGTSTP);
    sigaddset(&set, SIGCONT);
    pthread_sigmask(SIG_BLOCK, &set, NULL);
}

static void cr_cleanup(struct cr_ctx *ctx) {
    /* For now, just cancel the cosimulator thread.
     * It should be in pthread_cond_wait() and will go quickly.
     */

    pthread_cancel(ctx->thread);
    pthread_mutex_unlock(&ctx->mutex);
    pthread_cond_signal(&ctx->cosim_cond); // Make it run
    pthread_join(ctx->thread, NULL);       // Wait for it.
    pthread_cond_destroy(&ctx->spice_cond);
    pthread_cond_destroy(&ctx->cosim_cond);
    pthread_mutex_destroy(&ctx->mutex);
}

static void cr_yield_to_spice(struct cr_ctx *ctx) {
    pthread_cond_signal(&ctx->spice_cond);
    pthread_cond_wait(&ctx->cosim_cond, &ctx->mutex);
}
#endif /* pthread code. */
#endif // _COSIM_COROUTINE_SHIM_H_

