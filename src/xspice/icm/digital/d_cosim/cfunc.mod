/* Code model d_cosim.
 *
 * XSPICE code model for running a co-simulation with no support
 * for abandoning the current timestep.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#if defined (__MINGW32__) || defined (__CYGWIN__) || defined (_MSC_VER)
/* MS WINDOWS. */
#undef BOOLEAN
#include <windows.h>

#define dlopen(name, type) LoadLibrary(name)
#define dlsym(handle, name) (void *)GetProcAddress(handle, name)
#define dlclose(handle) FreeLibrary(handle)

char *dlerror(void) // Lifted from dev.c.
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
        snprintf(errstr, sizeof errstr, errstr_fmt, lpMsgBuf);
        LocalFree(lpMsgBuf);
    }
    return errstr;
} /* end of function dlerror */
#else
#include <dlfcn.h>
#endif

#include "ngspice/cosim.h"

/* The argument passed to code model functions. */

#define XSPICE_ARG      mif_private

#define DBG(...)
//#define DBG(...) cm_message_printf(__VA_ARGS__)

/* Structure used to hold queued inputs. */

struct pend_in {
    double          when;        // Due time.
    unsigned int    which;       // Index of input.
    Digital_t       what;        // The value.
};

/* Structure to maintain context, pointed to by STATIC VAR cosim_instance. */

struct instance {
    struct co_info  info;        // Co-simulation interface - MUST BE FIRST.
    int             q_index;     // Queue index (last active entry).
    unsigned int    q_length;    // Size of input queue.
    struct pend_in *q;           // The input queue.
    unsigned int    in_ports;    // Number of XSPICE inputs.
    unsigned int    out_ports;   // Number of XSPICE outputs.
    unsigned int    inout_ports; // Number of XSPICE inout ports.
    unsigned int    op_pending;  // Output is pending.
    Digital_t      *out_vals;    // The new output values.
    double          extra;       // Margin to extend timestep.
    void           *so_handle;   // dlopen() handle to the simulation binary.
};

/* Called at end of simulation run to free memory. */

static void callback(ARGS, Mif_Callback_Reason_t reason)
{
    struct instance *ip;

    ip = (struct instance *)STATIC_VAR(cosim_instance);
    if (reason == MIF_CB_DESTROY) {
        if (!ip)
            return;
        if (ip->info.cleanup)
            (*ip->info.cleanup)(&ip->info);
        if (ip->so_handle)
            dlclose(ip->so_handle);
        if (ip->q)
            free(ip->q);
        if (ip->out_vals)
            free(ip->out_vals);
        free(ip);
        STATIC_VAR(cosim_instance) = NULL;
    }
}

/* Function called when a co-simulator output changes.
 * Out-of-range values for bit_num must be ignored.
 */

void accept_output(struct co_info *pinfo, unsigned int bit_num, Digital_t *val)
{
    struct instance *ip = (struct instance *)pinfo; // First member.
    Digital_t       *out_vals;                      // XSPICE rotating memory.

    if (bit_num >= ip->out_ports + ip->inout_ports)
        return;
    out_vals = (Digital_t *)cm_event_get_ptr(1, 0);
    DBG("Change %s %d/%d->%d/%d vtime %g",
        cm_get_node_name("d_out", bit_num),
        out_vals[bit_num].state, out_vals[bit_num].strength,
        val->state, val->strength,
        ip->info.vtime);
    if (ip->op_pending == 0) {
        /* Prepare pending output. */

        memcpy(ip->out_vals, out_vals, ip->out_ports * sizeof *ip->out_vals);
        ip->op_pending = 1;
    }
    ip->out_vals[bit_num] = *val;
}

/* Push pending outputs, usually sent back from the future.
 * It is safe to use OUTPUT() here, although it mays seem that this
 * function may be called twice in a single call to cm_d_cosim().
 * There will never be any input changes when cm_d_cosim() is called
 * with pending output, as all input for the shortened time-step has
 * already been processed.
 */

static void output(struct instance *ip, ARGS)
{
    double        delay;
    Digital_t    *out_vals; // XSPICE rotating memory
    unsigned int  i, j;

    delay = PARAM(delay) - (TIME - ip->info.vtime);
    if (delay <= 0) {
        cm_message_printf("WARNING: output scheduled with impossible "
                          "delay (%g) at %g.", delay, TIME);
        delay = 1e-12;
    }
    out_vals = (Digital_t *)cm_event_get_ptr(1, 0);

    /* Output to d_out. */

    for (i = 0; i < ip->out_ports; ++i) {
        if (ip->out_vals[i].state != out_vals[i].state ||
            ip->out_vals[i].strength != out_vals[i].strength) {
            DBG("%g: OUT %s %d/%d->%d/%d vtime %g with delay %g",
                TIME, cm_get_node_name("d_out", i),
                out_vals[i].state, out_vals[i].strength,
                ip->out_vals[i].state, ip->out_vals[i].strength,
                ip->info.vtime, delay);
            *(Digital_t *)OUTPUT(d_out[i]) = out_vals[i] = ip->out_vals[i];
            OUTPUT_DELAY(d_out[i]) = delay;
            OUTPUT_CHANGED(d_out[i]) = TRUE;
        } else {
            OUTPUT_CHANGED(d_out[i]) = FALSE;
        }
    }

    /* Output to d_inout. */

    for (i = 0, j = ip->out_ports; i < ip->inout_ports; ++i, ++j) {
        if (ip->out_vals[j].state != out_vals[j].state ||
            ip->out_vals[j].strength != out_vals[j].strength) {
            DBG("%g: inOUT %s %d/%d->%d/%d vtime %g with delay %g",
                TIME, cm_get_node_name("d_inout", i),
                out_vals[j].state, out_vals[j].strength,
                ip->out_vals[j].state, ip->out_vals[j].strength,
                ip->info.vtime, delay);
            *(Digital_t *)OUTPUT(d_inout[i]) = out_vals[j] = ip->out_vals[j];
            OUTPUT_DELAY(d_inout[i]) = delay;
            OUTPUT_CHANGED(d_inout[i]) = TRUE;
        } else {
            OUTPUT_CHANGED(d_inout[i]) = FALSE;
        }
    }
    ip->op_pending = 0;
}

/* Run the co-simulation. Return 1 if the timestep was truncated. */

static int advance(struct instance *ip, ARGS)
{
    /* The co-simulator should advance to the time in ip->info.vtime,
     * but should pause when output is generated and update vtime.
     */

    (*ip->info.step)(&ip->info);

    if (ip->op_pending) {

        /* The co-simulator produced some output. */

        if (TIME - ip->info.vtime <= PARAM(delay)) {
#if 1
            DBG("Direct output with SPICE %.16g CS %.16g",
                TIME, ip->info.vtime);
            output(ip, XSPICE_ARG); // Normal output, unlikely.
#else
            cm_event_queue((TIME + ip->info.vtime + PARAM(delay)) / 2.0);
#endif
        } else {

            /* Something changed that may alter the future of the
             * SPICE simulation.  Truncate the current timestep so that
             * SPICE will see the pending output, which currently occurred
             * in the past.
             */

            DBG("Truncating timestep to %.16g", ip->info.vtime + ip->extra);
            cm_analog_set_temp_bkpt(ip->info.vtime + ip->extra);

            /* Any remaining input events are in an alternate future. */

            ip->q_index = -1;
            return 1;
        }
    }
    return 0;
}

/* Called from the main function to run the co-simulation. */

static void run(struct instance *ip, ARGS)
{
    struct pend_in  *rp;
    double           sim_started;
    int              i;

    if (ip->q_index < 0) {
        /* No queued input, advance to current TIME. */

        DBG("Advancing vtime without input %.16g -> %.16g",
            ip->info.vtime , TIME);
        ip->info.vtime = TIME;
        advance(ip, XSPICE_ARG);
        return;
    }

    /* Scan the queue. */

    DBG("%.16g: Running Q with %d entries", TIME, ip->q_index + 1);
    sim_started = ip->info.vtime;

    for (i = 0; i <= ip->q_index; ++i) {
        rp = ip->q + i;
        if (rp->when <= sim_started) {
            /* Not expected.  */

            cm_message_printf("Warning simulated event is in the past:\n"
                              "XSPICE %.16g\n"
                              "Event  %.16g\n"
                              "Cosim  %.16g",
                              TIME, rp->when, ip->info.vtime);
            cm_message_printf("i=%d index=%d", i, ip->q_index);
            continue;
        }

        /* Step the simulation forward to the input event time. */

        ip->info.vtime = rp->when;
        if (ip->info.method == Normal && advance(ip, XSPICE_ARG)) {
            ip->q_index = -1;
            return;
        }

        /* Pass input change to simulation. */

        (*ip->info.in_fn)(&ip->info, rp->which, &rp->what);
        while (i < ip->q_index && ip->q[i + 1].when == rp->when) {
            /* Another change at the same time. */

            ++i;
            rp = ip->q + i;
            (*ip->info.in_fn)(&ip->info, rp->which, &rp->what);
        }

        /* Simulator requested to run after input change. */

        if (ip->info.method == After_input && advance(ip, XSPICE_ARG)) {
            ip->q_index = -1;
            return;
        }
    }

    /* All input was processed.  Advance to end of the timestep. */

    ip->q_index = -1;
    if (ip->info.method == Normal && TIME > ip->info.vtime) {
        ip->info.vtime = TIME;
        advance(ip, XSPICE_ARG);
    }
}

/* Check whether an input value has changed.
 * To reduce the number of arguments, a struture pointer is passed.
 */

static bool check_input(struct instance *ip, Digital_t *ovp,
                        struct pend_in *rp)
{
    if (ovp->state != rp->what.state ||
        ovp->strength != rp->what.strength) {
        if (++ip->q_index < (int) ip->q_length) {
            /* Record this event. */

            ip->q[ip->q_index] = *rp;
        } else {
            /* Queue is full.  Handle that by forcing a shorter timestep. */

            --ip->q_index;
            while (ip->q_index >= 0 && ip->q[ip->q_index].when >= rp->when)
                --ip->q_index;
            if (ip->q_index >= 0) {
                cm_analog_set_temp_bkpt(
                    (rp->when + ip->q[ip->q_index].when) / 2);
            } else {
                /* This should never happen. */

                cm_message_printf("Error: Event queue overflow at %e.",
                                  rp->when);
            }
        }
        return true;
    }
    return false;
}

/* The code model's main function. */

void ucm_d_cosim(ARGS)
{
    struct instance *ip;
    Digital_t       *in_vals; // XSPICE rotating memory
    unsigned int     i;
    int              index;

    if (INIT) {
        unsigned int   ins, outs, inouts;
        unsigned int   alloc_size;
        void          *handle;
        void         (*ifp)(struct co_info *);
        char          *fn;

        /* Initialise outputs. Done early in case of failure. */

        outs = PORT_NULL(d_out) ? 0 : PORT_SIZE(d_out);
        for (i = 0; i < outs; ++i) {
            OUTPUT_STATE(d_out[i]) = ZERO;
            OUTPUT_STRENGTH(d_out[i]) = STRONG;
            OUTPUT_DELAY(d_out[i]) = PARAM(delay);
        }

        inouts = PORT_NULL(d_inout) ? 0 : PORT_SIZE(d_inout);
        for (i = 0; i < inouts; ++i) {
            OUTPUT_STATE(d_inout[i]) = ZERO;
            OUTPUT_STRENGTH(d_inout[i]) = STRONG;
            OUTPUT_DELAY(d_inout[i]) = PARAM(delay);
        }

        /* Load the shared library containing the co-simulator. */

        fn = PARAM(simulation);
        handle = dlopen(fn, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            cm_message_send("Failed to load simulation binary. "
                            "Try setting LD_LIBRARY_PATH.");
            cm_message_send(dlerror());
            return;
        }
        ifp = (void (*)(struct co_info *))dlsym(handle, "Cosim_setup");
        if (*ifp == NULL) {
            cm_message_printf("ERROR: no entry function in %s", fn);
            cm_message_send(dlerror());
            dlclose(handle);
            return;
        }

        /* Get the instance data and initialise it. */

        ip = (struct instance *)calloc(1, sizeof *ip);
        if (!ip)
            goto no_ip;
        ip->so_handle = handle;
        ip->info.vtime = 0.0;
        ip->info.out_fn = accept_output;
        CALLBACK = callback;

        /* Store the simulation interface information. */

        (*ifp)(&ip->info);

        /* Check lengths. */

        ins = PORT_NULL(d_in) ? 0 : PORT_SIZE(d_in);
        if (ins != ip->info.in_count) {
            cm_message_printf("Warning: mismatched XSPICE/co-simulator "
	                      "input counts: %d/%d.",
			      ins, ip->info.in_count);
        }
        if (outs != ip->info.out_count) {
            cm_message_printf("Warning: mismatched XSPICE/co-simulator "
	                      "output counts: %d/%d.",
			      outs, ip->info.out_count);
        }

        if (inouts != ip->info.inout_count) {
            cm_message_printf("Warning: mismatched XSPICE/co-simulator "
	                      "inout counts: %d/%d.",
			      inouts, ip->info.inout_count);
        }

        /* Create input queue and output buffer. */

        ip->q_index = -1;
        ip->q_length = PARAM(queue_size);
	ip->in_ports = ins;
	ip->out_ports = outs;
	ip->inout_ports = inouts;
        if (ins + inouts > ip->q_length) {
            cm_message_send("WARNING: Input queue size should be greater than "
                            "number of input ports. Size increased.");
            ip->q_length = ins + inouts + 16;
        }
        alloc_size = ip->q_length * sizeof (struct pend_in);
        ip->q = (struct pend_in *)malloc(alloc_size);
        if (!ip->q)
            goto no_q;
        ip->op_pending = 0;
        ip->out_vals = (Digital_t *)calloc(outs + inouts, sizeof (Digital_t));
        if (!ip->out_vals)
            goto no_out_vals;
        ip->extra = PARAM(delay) / 3; // FIXME?
        STATIC_VAR(cosim_instance) = ip;

        /* Allocate XSPICE rotating storage to track changes. */

        cm_event_alloc(0, (ins  + inouts) * sizeof (Digital_t));
        cm_event_alloc(1, (outs + inouts) * sizeof (Digital_t));

        /* Declare irreversible. */

	if (PARAM(irreversible) > 0)
            cm_irreversible(PARAM(irreversible));
        return;

        /* Handle malloc failures. */
    no_out_vals:
        free(ip->q);
    no_q:
        free(ip);
    no_ip:
        cm_message_send("No memory!");
        return;
    }

    ip = STATIC_VAR(cosim_instance);
    if (!ip) {
        unsigned int ports;

        /* Error state.  Do nothing at all. */

        ports = PORT_NULL(d_out) ? 0 : PORT_SIZE(d_out);
        for (i = 0; i < ports; ++i)
            OUTPUT_CHANGED(d_out[i]) = FALSE;
        ports = PORT_NULL(d_inout) ? 0 : PORT_SIZE(d_inout);
        for (i = 0; i < ports; ++i)
            OUTPUT_CHANGED(d_inout[i]) = FALSE;
        return;
    }
    in_vals = (Digital_t *)cm_event_get_ptr(0, 0);

    if (TIME == 0.0) {
        /* Starting, so inputs may be settling. */

        for (i = 0; i < ip->in_ports; ++i) {
            Digital_t ival;

            ival = *(Digital_t *)INPUT(d_in[i]);
            (*ip->info.in_fn)(&ip->info, i, &ival);
            in_vals[i] = ival;
        }

        for (i = 0; i < ip->out_ports; ++i)
            OUTPUT_CHANGED(d_out[i]) = FALSE;

        for (i = 0; i < ip->inout_ports; ++i) {
            Digital_t ival;

            ival = *(Digital_t *)INPUT(d_inout[i]);
            (*ip->info.in_fn)(&ip->info, i + ip->in_ports, &ival);
            in_vals[i + ip->in_ports] = ival;
            OUTPUT_CHANGED(d_inout[i]) = FALSE;
        }
        return;
    }

    if (CALL_TYPE == ANALOG) // Belt and braces
        return;

    /* Check for pending output. */

    if (ip->op_pending) {
        output(ip, XSPICE_ARG);
    } else {
        for (i = 0; i < ip->out_ports; ++i)
            OUTPUT_CHANGED(d_out[i]) = FALSE;
        for (i = 0; i < ip->inout_ports; ++i)
            OUTPUT_CHANGED(d_inout[i]) = FALSE;
    }

    /* Check TIME as it may have gone backwards after a failed time-step. */

    index = ip->q_index;
    while (index >= 0 && TIME < ip->q[index].when)
        --index;
    ip->q_index = index;

    if (CALL_TYPE == EVENT) {
        struct pend_in input;
        unsigned int   limit, max;

        /* New input is expected here. */

        input.when = TIME;

        limit = ip->info.in_count + ip->info.inout_count;
        max = limit < ip->in_ports ? limit : ip->in_ports;
        limit -= max;

        for (input.which = 0; input.which < max; ++input.which) {
            input.what = *(Digital_t *)INPUT(d_in[input.which]);
            if (check_input(ip, in_vals + input.which, &input)) {
                DBG("%.16g: IN %s %d/%d->%d/%d",
                    TIME, cm_get_node_name("d_in", input.which),
                    in_vals[input.which].state, in_vals[input.which].strength,
                    input.what.state, input.what.strength);
                in_vals[input.which] = input.what;
            }
        }

        if (limit > ip->inout_ports)
            limit = ip->inout_ports;
        for (i = 0; i < limit; ++i, ++input.which) {
            input.what = *(Digital_t *)INPUT(d_inout[i]);
            if (check_input(ip, in_vals + input.which, &input)) {
                DBG("%.16g: INout %s %d/%d->%d/%d",
                    TIME, cm_get_node_name("d_inout", i),
                    in_vals[input.which].state, in_vals[input.which].strength,
                    input.what.state, input.what.strength);
                in_vals[ip->in_ports + i] = input.what;
            }
        }
    } else if (CALL_TYPE == STEP_PENDING) {
        /* The current timestep succeeded.  Run the co-simulator code
         * forward, replaying any saved input events.
         */

        if (TIME <= ip->info.vtime)
            cm_message_printf("XSPICE time is behind vtime:\n"
                              "XSPICE %.16g\n"
                              "Cosim  %.16g",
                              TIME, ip->info.vtime);
        run(ip, XSPICE_ARG);
    }
}
