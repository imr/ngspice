/*============================================================================
FILE    EVTdequeue.c

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

    This file contains function EVTdequeue which removes any items on the
    output and instance queues with event times matching the specified
    simulation time.

INTERFACES

    void EVTdequeue(CKTcircuit *ckt, double time)

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


static void EVTdequeue_output(CKTcircuit *ckt, double time);
static void EVTdequeue_inst(CKTcircuit *ckt, double time);

static void EVTprocess_output(
    CKTcircuit  *ckt,
    int         output_index,
    void        *value);


/*
EVTdequeue

This function removes any items on the output and instance queues
with event times matching the specified simulation time.  EVTiter
is then called to determine which instances need to be called.

*/


void EVTdequeue(
    CKTcircuit  *ckt,      /* The circuit structure */
    double      time)      /* The event time of the events to dequeue */
{

    /* Take all items on output queue with matching time */
    /* and set changed flags in output queue */
    EVTdequeue_output(ckt, time);

    /* Take all items on inst queue with matching time */
    /* and set to_call flags in inst queue */
    EVTdequeue_inst(ckt, time);

}


/*
EVTdequeue_output

This function de-queues output events with times matching the
specified time.
*/

static void EVTdequeue_output(
    CKTcircuit  *ckt,          /* The circuit structure */
    double      time)          /* The event time of the events to dequeue */
{

    int         i;
    int         j;

    int         num_pending;
    int         index;
    int         output_index;

    double      next_time;
    double      event_time;

    Evt_Output_Queue_t  *output_queue;

    Evt_Output_Event_t  *output;
    Evt_Output_Event_t  **output_ptr;


    /* Get pointers for fast access */
    output_queue = &(ckt->evt->queue.output);

    /* Exit if nothing pending on output queue or if next_time */
    /* != specified time */
    if(output_queue->num_pending == 0)
        return;
    if(output_queue->next_time != time)
        return;

    /* Scan the list of outputs pending */
    num_pending = output_queue->num_pending;
    for(i = 0; i < num_pending; i++) {

        /* Get the index of the output */
        index = output_queue->pending_index[i];

        /* Get pointer to next event in queue at this index */
        output = *(output_queue->current[index]);

        /* If cleaned or event time does not match current time, skip */
        if(!output || output->event_time != time)
            continue;

        /* It must match, so pull the event from the queue and process it */
        EVTprocess_output(ckt, index, output->value);

        /* Move current to point to next non-removed item in list */
        output_ptr = &(output->next);
        output = *output_ptr;
        while(output) {
            if(! output->removed)
                break;
            output_ptr = &(output->next);
            output = *output_ptr;
        }
        output_queue->current[index] = output_ptr;

        /* Mark that this index in the queue has been modified */
        if(! output_queue->modified[index]) {
            output_queue->modified[index] = MIF_TRUE;
            output_queue->modified_index[(output_queue->num_modified)++] = index;
        }
    }


    /* Update/compact the pending list and update the next_time */
    next_time = 1e30;
    for(i = 0, j = 0; i < num_pending; i++) {
        output_index = output_queue->pending_index[i];
        output = *(output_queue->current[output_index]);
        /* If nothing in queue at last_step, remove this index from the pending list */
        if(! output) {
            output_queue->pending[output_index] = MIF_FALSE;
            (output_queue->num_pending)--;
        }
        /* else, keep the index and update the next time */
        else {
            output_queue->pending_index[j] = output_queue->pending_index[i];
            j++;
            event_time = output->event_time;
            if(event_time < next_time)
                next_time = event_time;
        }
    }
    output_queue->next_time = next_time;


}



/*
EVTdequeue_inst

This function de-queues instance events with times matching the
specified time.
*/


void EVTdequeue_inst(
    CKTcircuit  *ckt,    /* The circuit structure */
    double      time)    /* The event time of the events to dequeue */
{

    int         i;
    int         j;

    int         num_pending;
    int         index;
    int         inst_index;

    double      next_time;
    double      event_time;

    Evt_Inst_Queue_t  *inst_queue;

    Evt_Inst_Event_t  *inst;


    /* Get pointers for fast access */
    inst_queue = &(ckt->evt->queue.inst);

    /* Exit if nothing pending on inst queue or if next_time */
    /* != specified time */
    if(inst_queue->num_pending == 0)
        return;
    if(inst_queue->next_time != time)
        return;

    /* Scan the list of insts pending */
    num_pending = inst_queue->num_pending;
    for(i = 0; i < num_pending; i++) {

        /* Get the index of the inst */
        index = inst_queue->pending_index[i];

        /* Get pointer to next event in queue at this index */
        inst = *(inst_queue->current[index]);

        /* If cleaned or event time does not match current time, skip */
        if(!inst || inst->event_time != time)
            continue;

        /* It must match, so pull the event from the queue and process it */
        if(! inst_queue->to_call[index]) {
            inst_queue->to_call[index] = MIF_TRUE;
            inst_queue->to_call_index[(inst_queue->num_to_call)++] =
                    index;
        }

        /* Move current to point to next item in list */
        inst_queue->current[index] = &(inst->next);

        /* Mark that this index in the queue has been modified */
        if(! inst_queue->modified[index]) {
            inst_queue->modified[index] = MIF_TRUE;
            inst_queue->modified_index[(inst_queue->num_modified)++] = index;
        }
    }


    /* Update/compact the pending list and update the next_time */
    next_time = 1e30;
    for(i = 0, j = 0; i < num_pending; i++) {
        inst_index = inst_queue->pending_index[i];
        inst = *(inst_queue->current[inst_index]);
        /* If nothing in queue at last_step, remove this index from the pending list */
        if(! inst) {
            inst_queue->pending[inst_index] = MIF_FALSE;
            (inst_queue->num_pending)--;
        }
        /* else, keep the index and update the next time */
        else {
            inst_queue->pending_index[j] = inst_queue->pending_index[i];
            j++;
            event_time = inst->event_time;
            if(event_time < next_time)
                next_time = event_time;
        }
    }
    inst_queue->next_time = next_time;



}



/*
EVTprocess_output

This function processes a specified output after it is pulled
from the queue.
*/


static void EVTprocess_output(
    CKTcircuit  *ckt,          /* The circuit structure */
    int         output_index,  /* The index of the output to process */
    void        *value)        /* The output value */
{

    int                 num_outputs;
    int                 node_index;
    int                 udn_index;
    int                 output_subindex;

    Evt_Output_Info_t   **output_table;
    Evt_Node_Info_t     **node_table;

    Evt_Node_t          *rhs;
    Evt_Node_t          *rhsold;

    Evt_Output_Queue_t  *output_queue;

    Mif_Boolean_t       equal;


    output_table = ckt->evt->info.output_table;
    node_table = ckt->evt->info.node_table;

    node_index = output_table[output_index]->node_index;
    num_outputs = node_table[node_index]->num_outputs;
    udn_index = node_table[node_index]->udn_index;

    rhs = ckt->evt->data.node->rhs;
    rhsold = ckt->evt->data.node->rhsold;

    /* Determine if output is different from rhsold value */
    /* and copy it to rhs AND rhsold if so */
    /* This is somewhat inefficient, but that's the way */
    /* we have setup the structures (rhs and rhsold must match)... */
    if(num_outputs > 1) {
        output_subindex = output_table[output_index]->output_subindex;
        g_evt_udn_info[udn_index]->compare
                (value,
                rhsold[node_index].output_value[output_subindex],
                &equal);
        if(! equal) {
            g_evt_udn_info[udn_index]->copy
                    (value, rhs[node_index].output_value[output_subindex]);
            g_evt_udn_info[udn_index]->copy
                    (value, rhsold[node_index].output_value[output_subindex]);
        }
    }
    else {
        g_evt_udn_info[udn_index]->compare
                (value,
                rhsold[node_index].node_value,
                &equal);
        if(! equal) {
            g_evt_udn_info[udn_index]->copy
                    (value, rhs[node_index].node_value);
            g_evt_udn_info[udn_index]->copy
                    (value, rhsold[node_index].node_value);
        }
    }

    /* If different, put in changed list of output queue */
    if(! equal) {
        output_queue = &(ckt->evt->queue.output);
        if(! output_queue->changed[output_index]) {
            output_queue->changed[output_index] = MIF_TRUE;
            output_queue->changed_index[(output_queue->num_changed)++] =
                    output_index;
        }
    }
}
