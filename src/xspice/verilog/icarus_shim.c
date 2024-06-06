/*
 * Main file for the shared library ivlng.so that is used
 * by ngspice's d_cosim code model to connect an instance of
 * an Icarus Verilog simulation (libvvp.so).
 * Licensed on the same terms as Ngspice.
 *
 * Copyright (c) 2024 Giles Atkinson
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

/* The VVP code runs on its own stack, handled by cr_xxx() functions. */

#include "coroutine_shim.h"

#include "ngspice/cmtypes.h" // For Digital_t
#include "ngspice/cosim.h"


#ifndef NGSPICELIBDIR
#if defined(__MINGW32__) || defined(_MSC_VER)
#define NGSPICELIBDIR "C:\\Spice64\\lib\\ngspice"
#else
#define NGSPICELIBDIR "/usr/local/lib/ngspice"
#endif
#endif

/* This header file defines the external interface. It also contains an initial
 * comment that describes how this shared library is used.
 */

#include "icarus_shim.h"

/* Report fatal errors. */

static void fail(const char *what, int why)
{
    fprintf(stderr, "Icarus shim failed in function %s: %s.\n",
            what, strerror(why));
    abort();
}

static void input(struct co_info *pinfo, unsigned int bit, Digital_t *val)
{
    struct ng_vvp    *ctx = (struct ng_vvp *)pinfo->handle;
    struct ngvp_port *pp;
    int               count, a, b, dirty;

    /* Convert the value. */

    if (val->strength <= HI_IMPEDANCE && val->state != UNKNOWN) {
        a = val->state; // Normal - '0'/'1'.
        b = 0;
    } else if (val->strength == HI_IMPEDANCE) {
        a = 0;          // High impedance - 'z'.
        b = 1;
    } else {
        a = 1;          // Undefined - 'x'.
        b = 1;
    }

    /* Find the port. */

    if (bit >= pinfo->in_count) {
        bit -= pinfo->in_count;
        if (bit >= pinfo->inout_count)
            return;
        pp = ctx->ports + ctx->ins + ctx->outs; // Point at inouts.
        count = ctx->inouts;
    } else {
        pp = ctx->ports;
        count = ctx->ins;
    }

    while (count--) {
        if (pp[count].position <= bit)
            break;
    }
    pp = pp + count;
    bit -= pp->position;

    /* Check and update. */

    dirty = 0;
    bit = pp->bits - bit - 1; // Bit position for big-endian.
    a <<= bit;
    if (a ^ pp->previous.aval) {
        if (a)
            pp->previous.aval |= a;
        else
            pp->previous.aval &= ~(1 << bit);
        dirty = 1;
    }
    b <<= bit;
    if (b ^ pp->previous.bval) {
        if (b)
            pp->previous.bval |= b;
        else
            pp->previous.bval &= ~(1 << bit);
        dirty = 1;
    }

    if (dirty && !(pp->flags & IN_PENDING)) {
        pp->flags |= IN_PENDING;
        ++ctx->in_pending;
    }
}

/* Move the VVP simulation forward. */

static void step(struct co_info *pinfo)
{
    struct ng_vvp *ctx = (struct ng_vvp *)pinfo->handle;
    int            i;

    /* Let VVP run.  It will stop when it has caught up with SPICE time
     * (pinfo->vtime) or produced output.
     */

    cr_yield_to_sim(&ctx->cr_ctx);

    /* Check for output. */

    if (ctx->out_pending) {
        struct ngvp_port *pp;
        uint32_t          changed, mask;
        int               limit, i, bit;

        limit = ctx->outs + ctx->inouts;
        for (i = 0, pp = ctx->ports + ctx->ins; i < limit; ++i, ++pp) {
            if (!(pp->flags & OUT_PENDING))
                continue;

            pp->flags &= ~OUT_PENDING;
            changed = (pp->new.aval ^ pp->previous.aval) |
                      (pp->new.bval ^ pp->previous.bval);
            if (changed) {
                bit = pp->position;
                mask = 1 << (pp->bits - 1);
                while (changed) {
                    if (mask & changed) {
                        const Digital_t lv_vals[] =
                            { {ZERO, STRONG}, {ONE, STRONG},
                              {ZERO, HI_IMPEDANCE}, {UNKNOWN, STRONG} };
                        int a, b;

                        a = (pp->new.aval & mask) != 0;
                        b = (pp->new.bval & mask) != 0;
                        a += (b << 1);
                        pinfo->out_fn(pinfo, bit, (Digital_t *)lv_vals + a);
                        changed &= ~mask;
                    }
                    mask >>= 1;
                    ++bit;
                }
                pp->previous.aval = pp->new.aval;
                pp->previous.bval = pp->new.bval;
            }
            if (--ctx->out_pending == 0)
                break;
        }
        if (ctx->out_pending)
            abort();
    }
}

static void cleanup(struct co_info *pinfo)
{
    struct ng_vvp *ctx = (struct ng_vvp *)pinfo->handle;

    if (!ctx)
        return;

    /* Tell VVP to exit. */

    ctx->stop = 1;
    cr_yield_to_sim(&ctx->cr_ctx);
    cr_cleanup(&ctx->cr_ctx);
    free(ctx->ports);
    dlclose(ctx->vvp_handle);
    free(ctx);
    pinfo->handle = NULL;
}

/* Static variable and function for passing context from this library
 * to an instance of ivlng.vpi running in the VVP thread.
 * Get_ng_vvp() is called in the VVP thread and must synchronise.
 * XSPICE initialisation is single-threaded, so a static is OK.
 */

static struct ng_vvp *context;

struct ng_vvp *Get_ng_vvp(void)
{
    return context;
}

/* Thread start function runs the Verilog simulation. */

void *run_vvp(void *arg)
{
    static const char * const  fn_names[] = { VVP_FN_0, VVP_FN_1, VVP_FN_2,
                                              VVP_FN_3, VVP_FN_4, 0 };
    struct co_info            *pinfo = (struct co_info *)arg;
    struct vvp_ptrs            fns;
    void                     **fpptr;
    const char                *file;
    struct ng_vvp             *ctx;
    int                        i;

    cr_safety();        // Make safe with signals.

    /* Find the functions to be called in libvvp. */

    fpptr = (void **)&fns;
    for (i = 0; ; ++i, ++fpptr) {
        if (!fn_names[i])
            break;
        *fpptr = dlsym(context->vvp_handle, fn_names[i]);
        if (!*fpptr) {
            fprintf(stderr, "Icarus shim failed to find VVP function: %s.\n",
                    dlerror());
            abort();
        }
    }

    /* Start the simulation. */

    fns.add_module_path(".");
    file = (pinfo->lib_argc >= 3) ? pinfo->lib_argv[2] : NULL; // VVP log file.
    fns.init(file, pinfo->sim_argc, (char **)pinfo->sim_argv);
    fns.no_signals();

    /* The VPI file will usually be /usr/local/lib/ngspice/ivlng.vpi
     * or C:\Spice64\lib\ngspice\ivlng.vpi.
     */

    if (pinfo->lib_argc >= 2 && pinfo->lib_argv[1][0]) // Explicit VPI file.
        file = pinfo->lib_argv[1];
    else
#ifdef STAND_ALONE
        file = "./ivlng";
#else
        file = NGSPICELIBDIR "/ivlng";
#endif
    fns.load_module(file);
    fns.run(pinfo->sim_argv[0]);

    /* The simulation has finished.  Do nothing until destroyed. */

    ctx = (struct ng_vvp *)pinfo->handle;
    ctx->stop = 1;
    for (;;)
        cr_yield_to_spice(&ctx->cr_ctx);

    return NULL;
}

/* Entry point to this shared library.  Called by d_cosim. */

void Cosim_setup(struct co_info *pinfo)
{
    char             *file;
    struct ngvp_port *last_port;

    /* It is assumed that there is no parallel access to this function
     * as ngspice initialisation is single-threaded.
     */

    context = calloc(1, sizeof (struct ng_vvp));
    if (!context)
        fail("malloc", errno);
    context->cosim_context = pinfo;
    pinfo->handle = context;

    /* Load libvvp.  It is not statically linked as that would create
     * an unwanted build dependency on Icarus Verilog.
     */

    if (pinfo->lib_argc > 0 && pinfo->lib_argv[0][0]) // Explicit path to VVP?
        file = (char *)pinfo->lib_argv[0];
    else //libvvp is assumed to be in the OS search path.
        file = "libvvp";
    context->vvp_handle = pinfo->dlopen_fn(file);
    if (!context->vvp_handle) {
        fprintf(stderr, "Icarus shim failed to load VVP library\n");
        abort();
    }

    /* Set-up the execution stack for libvvp and start it. */

    cr_init(&context->cr_ctx, run_vvp, pinfo);

    /* Return required values in *pinfo. */

    last_port = context->ports + context->ins - 1;
    pinfo->in_count = context->ins ? last_port->position + last_port->bits : 0;
    last_port += context->outs;
    pinfo->out_count =
        context->outs ? last_port->position + last_port->bits : 0;
    last_port += context->inouts;
    pinfo->inout_count =
        context->inouts ? last_port->position + last_port->bits : 0;
    pinfo->cleanup = cleanup;
    pinfo->step = step;
    pinfo->in_fn = input;
    pinfo->method = Normal;
}
