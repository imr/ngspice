/* ===========================================================================
FILE    CM.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains functions callable from user code models.

INTERFACES

    cm_analog_alloc()
    cm_analog_get_ptr()
    cm_analog_integrate()
    cm_analog_converge()
    cm_analog_set_temp_bkpt()
    cm_analog_set_perm_bkpt()
    cm_analog_ramp_factor()
    cm_analog_not_converged()
    cm_analog_auto_partial()

    cm_message_get_errmsg()
    cm_message_send()
    cm_get_path()
    cm_get_circuit()

    cm_get_node_name()
    cm_probe_node()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */
#include "ngspice/ngspice.h"
#include "ngspice/cm.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"
#include "ngspice/enh.h"
#include "ngspice/mif.h"
#include "ngspice/cktdefs.h"
//#include "util.h"





static void cm_static_integrate(int byte_index,
                                double integrand,
                                double *integral,
                                double *partial);

/*

cm_analog_alloc()

This function is called from code model C functions to allocate
state storage for a particular instance.  It computes the number
of doubles that need to be allocated in SPICE's state storage
vectors from the number of bytes specified in it's argument and
then allocates space for the states.  An index into the SPICE
state-vectors is stored in the instance's data structure along
with a ``tag'' variable supplied by the caller so that the location
of the state storage area can be found by cm_analog_get_ptr().

*/

void cm_analog_alloc(
    int tag,            /* The user-specified tag for this block of memory */
    int bytes)          /* The number of bytes to allocate */
{
    MIFinstance *here;
    CKTcircuit  *ckt;

    Mif_State_t *state;

    int         doubles_needed;
    int         i;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;

    /* Scan states in instance struct and see if tag has already been used */
    for(i = 0; i < here->num_state; i++) {
        if(tag == here->state[i].tag) {
            g_mif_info.errmsg = "ERROR - cm_analog_alloc() - Tag already used in previous call\n";
            return;
        }
    }

    /* Compute number of doubles needed and allocate space in ckt->CKTstates[i] */
    doubles_needed = bytes / (int) sizeof(double) + 1;

    /* Allocate space in instance struct for this state descriptor */
    if(here->num_state == 0) {
        here->num_state = 1;
        here->state = TMALLOC(Mif_State_t, 1);
    }
    else {
        here->num_state++;
        here->state = TREALLOC(Mif_State_t, here->state, here->num_state);
    }

    /* Fill in the members of the state descriptor struct */
    state = &(here->state[here->num_state - 1]);
    state->tag = tag;
    state->index = ckt->CKTnumStates;
    state->doubles = doubles_needed;
    state->bytes = bytes;


    /* Add the states to the ckt->CKTstates vectors */
    ckt->CKTnumStates += doubles_needed;
    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
        if(ckt->CKTnumStates == doubles_needed)
            ckt->CKTstates[i] = TMALLOC(double, ckt->CKTnumStates);
        else
            ckt->CKTstates[i] = TREALLOC(double, ckt->CKTstates[i], ckt->CKTnumStates);
    }
}


/*
cm_analog_get_ptr()

This function is called from code model C functions to return a
pointer to state storage allocated with cm_analog_alloc().  A tag
specified in its argument list is used to locate the state in
question.  A second argument specifies whether the desired state
is for the current timestep or from a preceding timestep.  The
location of the state in memory is then computed and returned.
*/

void *cm_analog_get_ptr(
    int tag,            /* The user-specified tag for this block of memory */
    int timepoint)      /* The timepoint of interest - 0=current 1=previous */
{
    MIFinstance *here;
    CKTcircuit  *ckt;

    Mif_State_t *state=NULL;

    Mif_Boolean_t  got_tag;

    int         i;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;

    /* Scan states in instance struct and see if tag exists */
    for(got_tag = MIF_FALSE, i = 0; i < here->num_state; i++) {
        if(tag == here->state[i].tag) {
            state = &(here->state[i]);
            got_tag = MIF_TRUE;
            break;
        }
    }

    /* Return error if tag not found */
    if(! got_tag) {
        g_mif_info.errmsg = "ERROR - cm_analog_get_ptr() - Bad tag\n";
        return(NULL);
    }

    /* Return error if timepoint is not 0 or 1 */
    if((timepoint < 0) || (timepoint > 1)) {
        g_mif_info.errmsg = "ERROR - cm_analog_get_ptr() - Bad timepoint\n";
        return(NULL);
    }

    /* Return address of requested state in ckt->CKTstates[timepoint] vector */
    return( (void *) (ckt->CKTstates[timepoint] + state->index) );

}


/*
cm_analog_integrate()

This function performs a numerical integration on the state
supplied in its argument list according to the integrand also
supplied in the argument list.  The next value of the integral
and the partial derivative with respect to the integrand input is
returned.  The integral argument must be a pointer to memory
previously allocated through a call to cm_analog_alloc().  If this is
the first call to cm_analog_integrate(), information is entered into the
instance structure to mark that the integral should be processed
by MIFtrunc and MIFconvTest.
*/

int  cm_analog_integrate(
    double integrand,      /* The integrand */
    double *integral,      /* The current and returned value of integral */
    double *partial)       /* The partial derivative of integral wrt integrand */
{

    MIFinstance *here;
    CKTcircuit  *ckt;

    Mif_Intgr_t  *intgr;
    Mif_Boolean_t got_index;

    char        *char_state0;
    char        *char_state;

    int         byte_index;
    int         i;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;

    /* Check to be sure we're in transient analysis */
    if(g_mif_info.circuit.anal_type != MIF_TRAN) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_integrate() - Called in non-transient analysis\n";
        *partial  = 0.0;
        return(MIF_ERROR);
    }

    /* Preliminary check to be sure argument was allocated by cm_analog_alloc() */
    if(ckt->CKTnumStates <= 0) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_integrate() - Integral must be memory allocated by cm_analog_alloc()\n";
        *partial  = 0.0;
        return(MIF_ERROR);
    }

    /* Compute byte offset from start of state0 vector */
    char_state0 = (char *) ckt->CKTstate0;
    char_state  = (char *) integral;
    byte_index  = (int) (char_state - char_state0);

    /* Check to be sure argument address is in range of state0 vector */
    if((byte_index < 0) ||
       (byte_index > (ckt->CKTnumStates - 1) * (int) sizeof(double))) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_integrate() - Argument must be in state vector 0\n";
        *partial  = 0.0;
        return(MIF_ERROR);
    }

    /* Scan the intgr array in the instance struct to see if already exists */
    for(got_index = MIF_FALSE, i = 0; i < here->num_intgr; i++) {
        if(here->intgr[i].byte_index == byte_index) {
            got_index = MIF_TRUE;
        }
    }

    /* Report error if not found and this is not the first load pass in tran analysis */
    if((! got_index) && (! g_mif_info.circuit.anal_init)) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_integrate() - New integral and not initialization pass\n";
        *partial  = 0.0;
        return(MIF_ERROR);
    }

    /* If new integral state, allocate space in instance */
    /* struct for this intgr descriptor and register it with */
    /* the cm_analog_converge() function */
    if(! got_index) {
        if(here->num_intgr == 0) {
            here->num_intgr = 1;
            here->intgr = TMALLOC(Mif_Intgr_t, 1);
        }
        else {
            here->num_intgr++;
            here->intgr = TREALLOC(Mif_Intgr_t, here->intgr, here->num_intgr);
        }
        intgr = &(here->intgr[here->num_intgr - 1]);
        intgr->byte_index = byte_index;
        if(cm_analog_converge(integral)) {
            printf("%s\n",g_mif_info.errmsg);
            g_mif_info.errmsg = "ERROR - cm_analog_integrate() - Failure in cm_analog_converge() call\n";
            return(MIF_ERROR);
        }
    }

    /* Compute the new integral and the partial */
    cm_static_integrate(byte_index, integrand, integral, partial);

    return(MIF_OK);
}


/*
cm_analog_converge()

This function registers a state variable allocated with
cm_analog_alloc() to be subjected to a convergence test at the end of
each iteration.  The state variable must be a double. 
Information is entered into the instance structure to mark that
the state variable should be processed by MIFconvTest.
*/

int  cm_analog_converge(
    double *state)       /* The state to be converged */
{
    MIFinstance *here;
    CKTcircuit  *ckt;

    Mif_Conv_t  *conv;

    char        *char_state0;
    char        *char_state;

    int         byte_index;
    int         i;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;

    /* Preliminary check to be sure argument was allocated by cm_analog_alloc() */
    if(ckt->CKTnumStates <= 0) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_converge() - Argument must be memory allocated by cm_analog_alloc()\n";
        return(MIF_ERROR);
    }

    /* Compute byte offset from start of state0 vector */
    char_state0 = (char *) ckt->CKTstate0;
    char_state  = (char *) state;
    byte_index  = (int) (char_state - char_state0);

    /* Check to be sure argument address is in range of state0 vector */
    if((byte_index < 0) ||
       (byte_index > (ckt->CKTnumStates - 1) * (int) sizeof(double))) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_converge() - Argument must be in state vector 0\n";
        return(MIF_ERROR);
    }

    /* Scan the conv array in the instance struct to see if already registered */
    /* If so, do nothing, just return */
    for(i = 0; i < here->num_conv; i++) {
        if(here->conv[i].byte_index == byte_index)
            return(MIF_OK);
    }

    /* Allocate space in instance struct for this conv descriptor */
    if(here->num_conv == 0) {
        here->num_conv = 1;
        here->conv = TMALLOC(Mif_Conv_t, 1);
    }
    else {
        here->num_conv++;
        here->conv = TREALLOC(Mif_Conv_t, here->conv, here->num_conv);
    }

    /* Fill in the conv descriptor data */
    conv = &(here->conv[here->num_conv - 1]);
    conv->byte_index = byte_index;
    conv->last_value = 1.0e30;      /* There should be a better way ... */

    return(MIF_OK);
}



/*
cm_message_get_errmsg()

This function returns the address of an error message string set
by a call to some code model support function.
*/

char *cm_message_get_errmsg(void)
{
    return(g_mif_info.errmsg);
}




/*
cm_analog_set_temp_bkpt()

This function is called by a code model C function to set a
temporary breakpoint.  These temporary breakpoints remain in
effect only until the next timestep is taken.  A temporary
breakpoint added with a time less than the current time, but
greater than the last successful timestep causes the simulator to
abandon the current timestep and decrease the timestep to hit the
breakpoint.  A temporary breakpoint with a time greater than the
current time causes the simulator to make the breakpoint the next
timepoint if the next timestep would produce a time greater than
that of the breakpoint.
*/


int cm_analog_set_temp_bkpt(
    double time)              /* The time of the breakpoint to be set */
{
    CKTcircuit  *ckt;


    /* Get the address of the ckt and instance structs from g_mif_info */
    ckt  = g_mif_info.ckt;

    /* Make sure breakpoint is not prior to last accepted timepoint */
    if(time < ((ckt->CKTtime - ckt->CKTdelta) + ckt->CKTminBreak)) {
        g_mif_info.errmsg =
        "ERROR - cm_analog_set_temp_bkpt() - Time < last accepted timepoint\n";
        return(MIF_ERROR);
    }

    /* If too close to a permanent breakpoint or the current time, discard it */
    if ((ckt->CKTbreaks &&
         (fabs(time - ckt->CKTbreaks[0]) < ckt->CKTminBreak ||
          fabs(time - ckt->CKTbreaks[1]) < ckt->CKTminBreak)) ||
        fabs(time - ckt->CKTtime) < ckt->CKTminBreak) {
        return(MIF_OK);
    }

    /* If < current dynamic breakpoint, make it the current breakpoint */
    if( time < g_mif_info.breakpoint.current)
        g_mif_info.breakpoint.current = time;

    return(MIF_OK);
}




/*
cm_analog_set_perm_bkpt()

This function is called by a code model C function to set a
permanent breakpoint.  These permanent breakpoints remain in
effect from the time they are introduced until the simulation
time equals or exceeds the breakpoint time.  A permanent
breakpoint added with a time less than the current time, but
greater than the last successful timestep causes the simulator to
abandon the current timestep and decrease the timestep to hit the
breakpoint.  A permanent breakpoint with a time greater than the
current time causes the simulator to make the breakpoint the next
timepoint if the next timestep would produce a time greater than
that of the breakpoint.
*/


int cm_analog_set_perm_bkpt(
    double time)              /* The time of the breakpoint to be set */
{
    CKTcircuit  *ckt;


    /* Get the address of the ckt and instance structs from g_mif_info */
    ckt  = g_mif_info.ckt;

    /* Call cm_analog_set_temp_bkpt() to force backup if less than current time */
    if(time < (ckt->CKTtime + ckt->CKTminBreak))
        return(cm_analog_set_temp_bkpt(time));
    else
        CKTsetBreak(ckt,time);

    return(MIF_OK);
}


/*
cm_analog_ramp_factor()

This function returns the current value of the ramp factor
associated with the ``ramptime'' option.  For this option
to work best, models with analog outputs that may be non-zero at
time zero should call this function and scale their outputs
and partials by the ramp factor.
*/


double cm_analog_ramp_factor(void)
{

    CKTcircuit  *ckt;

    /* Get the address of the ckt and instance structs from g_mif_info */
    ckt  = g_mif_info.ckt;


    /* if ramptime == 0.0, no ramptime option given, so return 1.0 */
    /* this is the most common case, so it goes first */
    if(ckt->enh->ramp.ramptime == 0.0)
        return(1.0);

    /* else if not transient analysis, return 1.0 */
    else if( (!(ckt->CKTmode & MODETRANOP)) && (!(ckt->CKTmode & MODETRAN)) )
        return(1.0);

    /* else if time >= ramptime, return 1.0 */
    else if(ckt->CKTtime >= ckt->enh->ramp.ramptime)
        return(1.0);

    /* else time < end of ramp, so compute and return factor based on time */
    else
        return(ckt->CKTtime / ckt->enh->ramp.ramptime);
}


/* ************************************************************ */


/*
 * Copyright (c) 1985 Thomas L. Quarles
 *
 * This is a modified version of the function NIintegrate()
 *
 */

static void cm_static_integrate(int    byte_index,
                                double integrand,
                                double *integral,
                                double *partial)
{
    CKTcircuit  *ckt;

    double  intgr[7];
    double  cur=0;
    double  *double_ptr;

    double  ceq;
    double  geq;

    char    *char_ptr;

    int     i;


    /* Get the address of the ckt struct from g_mif_info */
    ckt  = g_mif_info.ckt;

    /* Get integral values from current and previous timesteps */
    for(i = 0; i <= ckt->CKTorder; i++) {
        char_ptr = (char *) ckt->CKTstates[i];
        char_ptr += byte_index;
        double_ptr = (double *) char_ptr;
        intgr[i] = *double_ptr;
    }


    /* Do what SPICE3C1 does for its implicit integration */

    switch(ckt->CKTintegrateMethod) {

    case TRAPEZOIDAL:

        switch(ckt->CKTorder) {

        case 1:
            cur = ckt->CKTag[1] * intgr[1];
            break;

        case 2:
            /* WARNING - This code needs to be redone.  */
            /* The correct code should rely on one previous value */
            /* of cur as done in NIintegrate() */
            cur = -0.5 * ckt->CKTag[0] * intgr[1];
            break;
        }

        break;

    case GEAR:
        cur = 0.0;

        switch(ckt->CKTorder) {

        case 6:
            cur += ckt->CKTag[6] * intgr[6];
            /* fall through */
        case 5:
            cur += ckt->CKTag[5] * intgr[5];
            /* fall through */
        case 4:
            cur += ckt->CKTag[4] * intgr[4];
            /* fall through */
        case 3:
            cur += ckt->CKTag[3] * intgr[3];
            /* fall through */
        case 2:
            cur += ckt->CKTag[2] * intgr[2];
            /* fall through */
        case 1:
            cur += ckt->CKTag[1] * intgr[1];
            break;

        }
        break;

    }

    ceq = cur;
    geq = ckt->CKTag[0];

    /* WARNING: Take this out when the case 2: above is fixed */
    if((ckt->CKTintegrateMethod == TRAPEZOIDAL) &&
       (ckt->CKTorder == 2))
        geq *= 0.5;


    /* The following code is equivalent to */
    /* the solution of one matrix iteration to produce the  */
    /* integral value.                                      */

    *integral = (integrand - ceq) / geq;
    *partial  = 1.0 / geq;

}





/*
cm_analog_not_converged()

This function tells the simulator not to allow the current
iteration to be the final iteration.  It is called when
a code model performs internal limiting on one or more of
its inputs to assist convergence.
*/

void cm_analog_not_converged(void)
{
    (g_mif_info.ckt->CKTnoncon)++;
}




/*
cm_message_send()

This function prints a message output from a code model, prepending
the instance name.
*/


int cm_message_send(
    char *msg)        /* The message to output. */
{
    MIFinstance *here;

    /* Get the address of the instance struct from g_mif_info */
    here = g_mif_info.instance;

    /* Print the name of the instance and the message */
    printf("\nInstance: %s   Message: %s\n", here->MIFname, msg);

    return(0);
}






/*
cm_analog_auto_partial()

This function tells the simulator to automatically compute
approximations of partial derivatives of analog outputs
with respect to analog inputs.  When called from a code
model, it sets a flag in the g_mif_info structure
which tells function MIFload() and it's associated
MIFauto_partial() function to perform the necessary
calculations.
*/


void cm_analog_auto_partial(void)
{
    g_mif_info.auto_partial.local = MIF_TRUE;
}

/*
cm_get_path()

Return the path of the first file given on the command line
after the command line options or set by the 'source' command.
Will be used in function fopen_with_path().
*/

char *cm_get_path(void)
{
    return Infile_Path;
}


/* cm_get_circuit(void)

To build complex custom-built xspice-models, access to certain
parameters (e.g. maximum step size) may be needed to get reasonable
results of a simulation. In detail, this may be necessary when
spice interacts with an external sensor-simulator and the results
of that external simulator do not have a direct impact on the spice
circuit. Then, modifying the maximum step size on the fly may help
to improve the simulation results. Modifying such parameters has to
be done carefully. The patch enhances the xspice interface with
access to the (fundamental) ckt pointer.
*/

CKTcircuit *cm_get_circuit(void)
{
    return(g_mif_info.ckt);
}

/* Get the name of a circuit node connected to a port. */

const char *cm_get_node_name(const char *port_name, unsigned int index)
{
    MIFinstance      *instance;
    Mif_Conn_Data_t  *conn;
    Mif_Port_Data_t  *port;
    int               i;

    instance = g_mif_info.instance;
    for (i = 0; i < instance->num_conn; ++i) {
        conn = instance->conn[i];
        if (!strcmp(port_name, conn->name)) {
            if (index >= (unsigned int)conn->size)
                return NULL;
            port = conn->port[index];
            if (port->type == MIF_DIGITAL || port->type == MIF_USER_DEFINED) {
                /* Event node, no name in port data. */

                i = port->evt_data.node_index;
                return g_mif_info.ckt->evt->info.node_table[i]->name;
            }
            return port->pos_node_str;
        }
    }
    return NULL;
}

/* Test the resolved value of a connected Digital/UDN node, given
 * an assumed value for a particular port.
 */

bool cm_probe_node(unsigned int  conn_index,  // Connection index
                   unsigned int  port_index,  // Port index within connection
                   void         *value)       // Inout UDN value
{
    MIFinstance      *instance;
    Mif_Conn_Data_t  *conn;
    Mif_Port_Data_t  *port;
    Mif_Evt_Data_t   *edata;
    Evt_Node_Info_t  *node_info;
    Evt_Node_t       *this;
    void             *hold;
    int               num_outputs;

    instance = g_mif_info.instance;
    if (conn_index >= (unsigned int)instance->num_conn)
        return FALSE;
    conn = instance->conn[conn_index];
    if (port_index >= (unsigned int)conn->size)
        return FALSE;
    port = conn->port[port_index];
    if (port->type != MIF_DIGITAL && port->type != MIF_USER_DEFINED)
        return FALSE;
    edata = &port->evt_data;
    node_info = g_mif_info.ckt->evt->info.node_table[edata->node_index];
    num_outputs = node_info->num_outputs;
    if (num_outputs <= 1)
        return num_outputs == 1;    // This should be the only output.
    this = g_mif_info.ckt->evt->data.node->rhsold + edata->node_index;

    /* Replace the actual output with the test value and resolve.
     * It is assumed that the resolve function will not use its output
     * as a working variable.  (True for digital, real and integer.)
     */

    hold = this->output_value[edata->output_subindex];
    this->output_value[edata->output_subindex] = value;
    g_evt_udn_info[node_info->udn_index]->resolve(num_outputs,
                                                  this->output_value,
                                                  value);
    this->output_value[edata->output_subindex] = hold;
    return TRUE;
}
