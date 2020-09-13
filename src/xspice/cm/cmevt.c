/* ===========================================================================
FILE    CMevt.c

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

    This file contains functions callable from user code models
    that are associated with the event-driven algorithm.

INTERFACES

    cm_event_alloc()
    cm_event_get_ptr()
    cm_event_queue()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"

#include "ngspice/cm.h"
#include "ngspice/mif.h"
#include "ngspice/evt.h"

#include "ngspice/evtproto.h"


/*
cm_event_alloc()

This function is called from code model C functions to allocate
state storage for a particular event-driven
instance. It is similar to the
function cm_analog_alloc() used by analog models, but allocates states
that are rotated during event-driven 'timesteps' instead of analog
timesteps.
*/


void cm_event_alloc(
    int tag,           /* The user-specified tag for the memory block */
    int bytes)         /* The number of bytes to be allocated */
{

    int         inst_index;
    int         num_tags;

    MIFinstance *here;
    CKTcircuit  *ckt;

    Evt_State_Desc_t    **desc_ptr;
    Evt_State_Desc_t    *desc;

    Evt_State_Data_t    *state_data;
    Evt_State_t         *state;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;


    /* If not initialization pass, return error */
    if(here->initialized) {
        g_mif_info.errmsg =
        "ERROR - cm_event_alloc() - Cannot alloc when not initialization pass\n";
        return;
    }


    /* Get pointers for fast access */
    inst_index = here->inst_index;
    state_data = ckt->evt->data.state;


    /* Scan state descriptor list to determine if tag is present and to */
    /* find the end of the list.  Report error if duplicate tag */
    desc_ptr = &(state_data->desc[inst_index]);
    desc = *desc_ptr;
    num_tags = 1;
    while(desc) {
        if(desc->tag == tag) {
            g_mif_info.errmsg =
            "ERROR - cm_event_alloc() - Duplicate tag\n";
            return;
        }
        desc_ptr = &(desc->next);
        desc = *desc_ptr;
        num_tags++;
    }

    /* Create a new state description structure at end of list */
    /* and fill in the data and update the total size */
    *desc_ptr = TMALLOC(Evt_State_Desc_t, 1);
    desc = *desc_ptr;
    desc->tag = tag;
    desc->size = bytes;
    desc->offset = state_data->total_size[inst_index];
    state_data->total_size[inst_index] += bytes;

    /* Create a new state structure if list starting at head is null */
    state = state_data->head[inst_index];
    if(state == NULL) {
        state = TMALLOC(Evt_State_t, 1);
        state_data->head[inst_index] = state;
    }

    /* Create or enlarge the block and set the time */
    if(num_tags == 1)
        state->block = tmalloc((size_t) state_data->total_size[inst_index]);
    else
        state->block = trealloc(state->block,
                             (size_t) state_data->total_size[inst_index]);

    state->step = g_mif_info.circuit.evt_step;
}





/*
cm_event_get_ptr()

This function is called from code model C functions to return a
pointer to state storage allocated with cm_event_alloc().  A tag
specified in its argument list is used to locate the state in
question.  A second argument specifies whether the desired state
is for the current timestep or from a preceding timestep.  The
location of the state in memory is then computed and returned.
*/


void *cm_event_get_ptr(
    int tag,            /* The user-specified tag for the memory block */
    int timepoint)      /* The timepoint - 0=current, 1=previous */
{

    int         i;
    int         inst_index;

    MIFinstance *here;
    CKTcircuit  *ckt;

    void        *ptr;

    Evt_State_Desc_t    *desc;

    Evt_State_Data_t    *state_data;
    Evt_State_t         *state;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;


    /* If initialization pass, return error */
    if((! here->initialized) && (timepoint > 0)) {
        g_mif_info.errmsg =
        "ERROR - cm_event_get_ptr() - Cannot get_ptr(tag,1) during initialization pass\n";
        return(NULL);
    }

    /* Get pointers for fast access */
    inst_index = here->inst_index;
    state_data = ckt->evt->data.state;

    /* Scan state descriptor list to find the descriptor for this tag. */
    /* Report error if tag not found */
    desc = state_data->desc[inst_index];
    while(desc) {
        if(desc->tag == tag)
            break;
        desc = desc->next;
    }

    if(desc == NULL) {
        g_mif_info.errmsg =
        "ERROR - cm_event_get_ptr() - Specified tag not found\n";
        return(NULL);
    }

    /* Get the state pointer from the current array */
    state = *(state_data->tail[inst_index]);

    /* Backup the specified number of timesteps */
    for(i = 0; i < timepoint; i++)
        if(state->prev)
            state = state->prev;

    /* Return pointer */
    ptr = ((char *) state->block) + desc->offset;
    return(ptr);
}




/*
cm_event_queue()

This function queues an event for an instance participating
in the event-driven algorithm.
*/


int  cm_event_queue(
    double time)       /* The time of the event to be queued */
{

    MIFinstance *here;
    CKTcircuit  *ckt;


    /* Get the address of the ckt and instance structs from g_mif_info */
    here = g_mif_info.instance;
    ckt  = g_mif_info.ckt;

    /* If breakpoint time <= current event time, return error */
    if(time <= g_mif_info.circuit.evt_step) {
        g_mif_info.errmsg =
        "ERROR - cm_event_queue() - Event time cannot be <= current time\n";
        return(MIF_ERROR);
    }

    /* Add the event time to the inst queue */
    EVTqueue_inst(ckt, here->inst_index, g_mif_info.circuit.evt_step,
                  time);

    return(MIF_OK);
}
