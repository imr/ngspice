/*============================================================================
FILE    EVTqueue.c

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

    This file contains functions that place new events into the output and
    instance queues.

INTERFACES

    void EVTqueue_output(
        CKTcircuit *ckt,
        int        output_index,
        int        udn_index,
        Evt_Output_Event_t  *new_event,
        double     posted_time,
        double     event_time)

    void EVTqueue_inst(
        CKTcircuit *ckt,
        int        inst_index,
        double     posted_time,
        double     event_time)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
//#include "util.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"



/*
EVTqueue_output

This function places the specified output event onto the output
queue.  It is called by EVTload during a transient analysis.

The linked list in the queue for the specified output is
searched beginning at the current head of the pending events
to find the location at which to insert the new event.  The
events are ordered in the list by event_time.  If the event
is placed before the end of the list, subsequent events are
removed from the list by marking them as 'removed' and
recording the time of removal.  This allows efficient backup
of the state of the queue if a subsequent analog timestep
fails.
*/


void EVTqueue_output(
    CKTcircuit *ckt,                 /* The circuit structure */
    int        output_index,         /* The output in question */
    int        udn_index,            /* The associated user-defined node type */
    Evt_Output_Event_t  *new_event,  /* The event to queue */
    double     posted_time,          /* The current time */
    double     event_time)           /* The time the event should happen */
{

    Evt_Output_Queue_t  *output_queue;
    Evt_Output_Event_t  **here;
    Evt_Output_Event_t  *next;

    Mif_Boolean_t       splice;

    NG_IGNORE(udn_index);

    /* Get pointers for fast access */
    output_queue = &(ckt->evt->queue.output);

    /* Put the times into the event struct */
    new_event->event_time = event_time;
    new_event->posted_time = posted_time;
    new_event->removed = MIF_FALSE;

    /* Update next_time in output queue */
    if((output_queue->num_pending <= 0) ||
            (event_time < output_queue->next_time))
        output_queue->next_time = event_time;

    /* Find location at which to insert event */
    splice = MIF_FALSE;
    here = output_queue->current[output_index];
    while(*here) {
        if(event_time <= (*here)->event_time) {
            splice = MIF_TRUE;
            break;
        }
        here = &((*here)->next);
    }

    /* If needs to be spliced into middle of existing list */
    if(splice) {
        /* splice it in */
        next = *here;
        *here = new_event;
        new_event->next = next;
        /* mark later events as removed */
        while(next) {
            if(! next->removed) {
                next->removed = MIF_TRUE;
                next->removed_time = posted_time;
            }
            next = next->next;
        }
    }
    /* else, just put it at the end */
    else {
        *here = new_event;
        new_event->next = NULL;
    }

    /* Add to the list of outputs modified since last accepted timestep */
    if(! output_queue->modified[output_index]) {
        output_queue->modified[output_index] = MIF_TRUE;
        output_queue->modified_index[(output_queue->num_modified)++] =
                output_index;
    }

    /* Add to the list of outputs with events pending */
    if(! output_queue->pending[output_index]) {
        output_queue->pending[output_index] = MIF_TRUE;
        output_queue->pending_index[(output_queue->num_pending)++] =
                output_index;
    }
}



/*
EVTqueue_inst

This function places the specified inst event onto the inst
queue.

The linked list in the queue for the specified inst is
searched beginning at the current head of the pending events
to find the location at which to insert the new event.  The
events are ordered in the list by event_time.
*/




void EVTqueue_inst(
    CKTcircuit *ckt,          /* The circuit structure */
    int        inst_index,    /* The instance in question */
    double     posted_time,   /* The current time */
    double     event_time)    /* The time the event should happen */
{

    Evt_Inst_Queue_t  *inst_queue;
    Evt_Inst_Event_t  **here;
    Evt_Inst_Event_t  *new_event;
    Evt_Inst_Event_t  *next;

    Mif_Boolean_t       splice;


    /* Get pointers for fast access */
    inst_queue = &(ckt->evt->queue.inst);

    /* Update next_time in inst queue */
    if((inst_queue->num_pending <= 0) ||
            (event_time < inst_queue->next_time))
        inst_queue->next_time = event_time;

    /* Find location at which to insert event */
    splice = MIF_FALSE;
    here = inst_queue->current[inst_index];
    while(*here) {
        /* If there's an event with the same time, don't duplicate it */
        if(event_time == (*here)->event_time) {
            return;
        }    
        else if(event_time < (*here)->event_time) {
            splice = MIF_TRUE;
            break;
        }
        here = &((*here)->next);
    }

    /* Create a new event or get one from the free list and copy in data */
    if(inst_queue->free[inst_index]) {
        new_event = inst_queue->free[inst_index];
        inst_queue->free[inst_index] = new_event->next;
    }
    else {
        new_event = TMALLOC(Evt_Inst_Event_t, 1);
    }
    new_event->event_time = event_time;
    new_event->posted_time = posted_time;

    /* If needs to be spliced into middle of existing list */
    if(splice) {
        /* splice it in */
        next = *here;
        *here = new_event;
        new_event->next = next;
    }
    /* else, just put it at the end */
    else {
        *here = new_event;
        new_event->next = NULL;
    }

    /* Add to the list of insts modified since last accepted timestep */
    if(! inst_queue->modified[inst_index]) {
        inst_queue->modified[inst_index] = MIF_TRUE;
        inst_queue->modified_index[(inst_queue->num_modified)++] =
                inst_index;
    }

    /* Add to the list of insts with events pending */
    if(! inst_queue->pending[inst_index]) {
        inst_queue->pending[inst_index] = MIF_TRUE;
        inst_queue->pending_index[(inst_queue->num_pending)++] =
                inst_index;
    }
}
