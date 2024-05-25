/*
 * Copyright (c) 2002 Stephen Williams (steve@icarus.com)
 * Copyright (c) 2023 Giles Atkinson
 *
 *    This source code is free software; you can redistribute it
 *    and/or modify it in source code form under the terms of the GNU
 *    General Public License as published by the Free Software
 *    Foundation; either version 2 of the License, or (at your option)
 *    any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
//#include <vpi_user.h>
#include "vpi_user_dummy.h"

#include "ngspice/cmtypes.h" // For Digital_t
#include "ngspice/cosim.h"

/* The VVP code runs on its own stack, handled by cr_xxx() functions. */

#include "coroutine_cosim.h"

/* This header file defines external interfaces. It also contains an initial
 * comment that describes how this VPI module is used.
 */

#include "icarus_shim.h"

/* Functions that VPI calls have a returned parameter argument that
 * is not used here.
 */

#define UNUSED __attribute__((unused))

/* Debugging printfs(). */

//#define DEBUG
#ifdef DEBUG
#define DBG(...) vpi_printf(__VA_ARGS__)
#else
#define DBG(...)
#endif

static PLI_INT32 next_advance_cb(struct t_cb_data *cb);

/* Get current simulation time: no module-specific values. */

static double get_time(struct ng_vvp *ctx)
{
    static struct t_vpi_time now = { .type = vpiSimTime };
    uint64_t                 ticks;

    vpi_get_time(NULL, &now);
    ticks = ((uint64_t)now.high << 32) + now.low;
    return (double)ticks * ctx->tick_length;
}

/* Arrange for end_advance_cb() to be called in the future. */

static vpiHandle set_stop(uint64_t length, struct ng_vvp *ctx)
{
    static struct t_vpi_time now = { .type = vpiSimTime };
    static struct t_cb_data  cbd = { .cb_rtn = next_advance_cb, .time = &now };

    now.low = (unsigned int)length;
    now.high = (unsigned int)(length >> 32);
    if (length == 0)
        cbd.reason = cbReadWriteSynch;
    else
        cbd.reason = cbAfterDelay;

    /* Callback after delay. */

    cbd.user_data = (PLI_BYTE8 *)ctx;
    return vpi_register_cb(&cbd);
}

/* Timed callback at end of simulation advance: wait for a command
 * from the main thread, and schedule the next callback.
 * On return, VVP runs some more.
 */

static PLI_INT32 next_advance_cb(struct t_cb_data *cb)
{
    struct ng_vvp *ctx = (struct ng_vvp *)cb->user_data;
    struct t_vpi_value val;
    double             vl_time;
    uint64_t           ticks;
    unsigned int       i;

    for (;;) {
        /* Still wanted? */

        if (ctx->stop) {
            vpi_control(vpiFinish, 0); // Returns after scheduling $finish.
            return 0;
        }

        /* Save base time for next slice. */

        ctx->base_time = ctx->cosim_context->vtime;

        /* Repeatedly wait for instructions from the main thread
         * until VVP can advance at least one time unit.
         */

        cr_yield_to_spice(&ctx->cr_ctx);

        /* Check for input. */

        val.format = vpiVectorVal;
        i = ctx->ins ? 0 : ctx->outs;
        while (ctx->in_pending) {
            if (ctx->ports[i].flags & IN_PENDING) {
                ctx->ports[i].flags ^= IN_PENDING;
                val.value.vector =
                    (struct t_vpi_vecval *)&ctx->ports[i].previous;
                vpi_put_value(ctx->ports[i].handle, &val, NULL, vpiNoDelay);
                ctx->in_pending--;
                DBG("VPI input %d/%d on %s\n",
                    val.value.vector->aval, val.value.vector->bval,
                    vpi_get_str(vpiName, ctx->ports[i].handle));
            } else if (++i == ctx->ins) {
                i = ctx->ins + ctx->outs;      // Now scan inouts
            }
        }

        /* How many VVP ticks to advance? */

        vl_time = get_time(ctx);
        if (ctx->cosim_context->vtime < vl_time) {
            /* This can occur legitimately as the two times need not
             * align exactly.  But it should be less than one SPICE timestep.
             */

            DBG("VVP time %.16g ahead of SPICE target %.16g\n",
                    vl_time, ctx->cosim_context->vtime);
            if (ctx->cosim_context->vtime + ctx->tick_length < vl_time) {
                fprintf(stderr,
                        "Error: time reversal (%.10g->%.10g) in "
                        "Icarus Shim VPI!\n",
                        vl_time, ctx->cosim_context->vtime);
            }
            continue;
        }

        ticks = (uint64_t)
            ((ctx->cosim_context->vtime - vl_time)  / ctx->tick_length);
        if (ticks > 0) {
            DBG("Advancing from %g to %g: %lu ticks\n",
                vl_time, ctx->cosim_context->vtime, ticks);
            ctx->stop_cb = set_stop(ticks, ctx);
            return 0;
        }
    }
}

/* Callback function - new output value. */

static PLI_INT32 output_cb(struct t_cb_data *cb)
{
    struct ngvp_port *pp = (struct ngvp_port *)cb->user_data;
    struct ng_vvp    *ctx = pp->ctx;

    DBG("Output: %s is %d now %g (VL) %g (SPICE)\n",
        vpi_get_str(vpiName, cb->obj),
        cb->value->value.vector->aval,
        get_time(ctx), ctx->cosim_context->vtime);

    if (ctx->stop_cb) {
        /* First call in the current VVP cycle: cancel the current
         * stop CB and request a new one before the next VVP time point.
         * That allows all output events in the current timestep
         * to be gathered before stopping.
         */

        vpi_remove_cb(ctx->stop_cb);
        ctx->stop_cb = NULL;
        set_stop(0, ctx);

        /* Set the output time in SPICE format.
         * It must not be earlier than entry time.
         */

        ctx->cosim_context->vtime = get_time(ctx);
        if (ctx->cosim_context->vtime < ctx->base_time)
            ctx->cosim_context->vtime = ctx->base_time;
    }

    /* Record the value. */

    pp->new.aval = cb->value->value.vector->aval;
    pp->new.bval = cb->value->value.vector->bval;
    if (!(pp->flags & OUT_PENDING)) {
        pp->flags |= OUT_PENDING;
        ++ctx->out_pending;
    }
    return 0;
}

/* Utilty functions for start_cb() - initialise or set watch on a variable.*/

static void init(vpiHandle handle)
{
    static struct t_vpi_value val = { .format = vpiIntVal };

    DBG("Initialising %s to 0\n", vpi_get_str(vpiName, handle));

    vpi_put_value(handle, &val, NULL, vpiNoDelay);
}

static void watch(vpiHandle handle, void *pp)
{
    static struct t_vpi_time  time = { .type = vpiSuppressTime };
    static struct t_vpi_value val = { .format = vpiVectorVal };
    static struct t_cb_data   cb = {
        .reason = cbValueChange, .cb_rtn = output_cb,
        .time = &time, .value = &val
    };

    cb.obj = handle;
    cb.user_data = pp;
    vpi_register_cb(&cb);
}

/* Callback function - simulation is starting. */

static PLI_INT32 start_cb(struct t_cb_data *cb)
{
    struct ng_vvp *ctx = (struct ng_vvp *)cb->user_data;
    vpiHandle      iter, top, item;
    PLI_INT32      direction;
    char          *name;
    int            ii, oi, ioi;

    DBG("Unit %d precision %d\n",
        vpi_get(vpiTimeUnit, NULL),  vpi_get(vpiTimePrecision, NULL));
    ctx->tick_length = pow(10.0, vpi_get(vpiTimeUnit, NULL));

    /* Find the (unique?) top-level module and the one inside it. */

    iter = vpi_iterate(vpiModule, NULL);
    top = vpi_scan(iter);
    vpi_free_object(iter);
    DBG("Top %s\n", vpi_get_str(vpiName, top));

    /* Count the ports. */

    iter = vpi_iterate(vpiPort, top);
    if (!iter)
        vpi_printf("Top module has no ports!\n"); // vpi_scan() aborts.
    ctx->ins = ctx->outs = ctx->inouts = 0;
    while ((item = vpi_scan(iter))) {
        direction = vpi_get(vpiDirection, item);
        switch (direction) {
        case vpiInput:
            ++ctx->ins;
            break;
        case vpiOutput:
            ++ctx->outs;
            break;
        case vpiInout:
            ++ctx->inouts;
            break;
        default:
            break;
        }
    }
    ctx->ports = (struct ngvp_port *)malloc(
                     (ctx->ins + ctx->outs + ctx->inouts) *
                     sizeof (struct ngvp_port));
    if (!ctx->ports) {
        vpi_printf("No memory for ports at " __FILE__ ":%d\n", __LINE__);
        abort();
    }

    /* Get the ports. */

    iter = vpi_iterate(vpiPort, top);
    ii = oi = ioi = 0;
    while ((item = vpi_scan(iter))) {
        vpiHandle         namesake;
        struct ngvp_port *pp;
        int               first;

        direction = vpi_get(vpiDirection, item);
        name = vpi_get_str(vpiName, item);

        /* Assume that there is an object with the same name as the port
         * whose value may be read or set as needed.  This is an assumption
         * about how code for VVP is generated.
         */

        namesake = vpi_handle_by_name(name, top);
        DBG("Port %s direction %d size %d, namesake type %d\n",
            name, direction, vpi_get(vpiSize, item),
            vpi_get(vpiType, namesake));

        switch (direction) {
        case vpiInput:
            first = !ii;
            pp = ctx->ports + ii++;
            init(namesake);
            break;
        case vpiOutput:
            first = !oi;
            pp = ctx->ports + ctx->ins + oi++;
            watch(namesake, pp);
            break;
        case vpiInout:
            first = !ioi;
            init(namesake);
            pp = ctx->ports + ctx->ins + ctx->outs + ioi++;
            watch(namesake, pp);
            break;
        default:
            continue;
        }
        pp->bits = (uint16_t)vpi_get(vpiSize, item);
        pp->flags = 0;
        pp->position = first ? 0 : pp[-1].position + pp[-1].bits;
        pp->previous.aval = pp->previous.bval = 0;
        pp->handle = namesake;
        pp->ctx = ctx;
    }

    /* Make a direct call to the "end-of-advance" callback to start running. */

    cr_init(&ctx->cr_ctx);
    cb->user_data = (PLI_BYTE8 *)ctx;
    next_advance_cb(cb);
    return 0;
}

/* VPI initialisation. */

static void start(void)
{
    static struct t_vpi_time now = { .type = vpiSimTime };
    static struct t_cb_data  cbd = { .reason = cbStartOfSimulation,
                                     .time = &now, .cb_rtn = start_cb };
#ifdef DEBUG
    struct t_vpi_vlog_info   info;

    /* Get the program name. */

    if (vpi_get_vlog_info(&info)) {
        vpi_printf("Starting ivlng.vpi in %s\n", info.argv[0]);
        for (int i = 0; i < info.argc; ++i)
            vpi_printf("%d: %s\n", i, info.argv[i]);
        vpi_printf("P: %s V: %s\n", info.product, info.version);
    } else {
        vpi_printf("Failed to get invocation information.\n");
    }
#endif

    /* The first step is to find the top-level module and query its ports.
     * At this point they do not exist, so request a callback once they do.
     */

    cbd.user_data = (PLI_BYTE8 *)Get_ng_vvp();
    vpi_register_cb(&cbd);
}

/* This is a table of registration functions. It is the external symbol
 * that the VVP simulator looks for when loading this .vpi module.
 */

void (*vlog_startup_routines[])(void) = {
      start,
      0
};
