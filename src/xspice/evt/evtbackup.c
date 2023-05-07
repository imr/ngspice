/*============================================================================
FILE    EVTbackup.c

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

    This file contains a function that resets the queues and data
    structures to their state at the new analog simulation time specified
    following the rejection of an analog timestep by the DCtran routine.

INTERFACES

    void EVTbackup(CKTcircuit  *ckt, double new_time)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


/*=== INCLUDE FILES ===*/
#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
//#include "util.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"

#include "ngspice/evtproto.h"



/*=== FUNCTION PROTOTYPES ===*/


static void EVTbackup_node_data(CKTcircuit  *ckt, double new_time);
static void EVTbackup_state_data(CKTcircuit  *ckt, double new_time);
static void EVTbackup_msg_data(CKTcircuit  *ckt, double new_time);
static void EVTbackup_inst_queue(CKTcircuit  *ckt, double new_time);
static void EVTbackup_output_queue(CKTcircuit  *ckt, double new_time);



/*
EVTbackup()

This function resets the queues and data structures to their state
at the new analog simulation time specified.  The algorithms in this file
assume the following timestep coordination between
analog and event-driven algorithms:

    while(not end of analysis) {

        while (next event time <= next analog time) {
            do event solution with call_type = event_driven
            if any instance set analog breakpoint < next analog time
                set next analog time to breakpoint
        }

        do analog timestep solution with call_type = analog
        call all hybrid models with call_type = event_driven

        if(analog solution doesn't converge)
            Call EVTbackup
        else
            Call EVTaccept
    }
*/


void EVTbackup(
    CKTcircuit  *ckt,       /* the main circuit structure */
    double      new_time)   /* the time to backup to */
{


    /* Backup the node data */
    EVTbackup_node_data(ckt, new_time);

    /* Backup the state data */
    EVTbackup_state_data(ckt, new_time);

    /* Backup the msg data */
    EVTbackup_msg_data(ckt, new_time);

    /* Backup the inst queue */
    EVTbackup_inst_queue(ckt, new_time);

    /* Backup the output queue */
    EVTbackup_output_queue(ckt, new_time);

    /* Record statistics */
    (ckt->evt->data.statistics->tran_time_backups)++;

} /* EVTbackup */




/*
EVTbackup_node_data()

Reset the node structure data.
*/


static void EVTbackup_node_data(
    CKTcircuit  *ckt,           /* the main circuit structure */
    double      new_time)       /* the time to backup to */
{

    int         i;
    int         j;

    int         num_modified;
    int         node_index;

    Evt_Node_Info_t     **node_table;
    Evt_Node_Data_t     *node_data;
    Evt_Node_t          **node_ptr;
    Evt_Node_t          *node;
    Evt_Node_t          *from_node;
    Evt_Node_t          *to_node;
    Evt_Node_t          *head;
    Evt_Node_t          *tail;
    Evt_Node_t          *free_head;

    /* Get pointers for quick access */
    node_data = ckt->evt->data.node;
    node_table = ckt->evt->info.node_table;

    /* Loop through list of indexes modified since last accepted timepoint */
    num_modified = node_data->num_modified;
    for(i = 0; i < num_modified; i++) {

        /* Get the needed node and udn indexes */
        node_index = node_data->modified_index[i];

        /* Scan data for this node from last_step to determine new setting */
        /* for tail, and splice later data into the free list */
        node_ptr = node_data->last_step[node_index];
        node = *node_ptr;
        for (;;) {
            if((node->next == NULL) || (node->next->step > new_time)) {

                /* Splice rest of list, if any, into free list */
                head = node->next;
                if(head) {
                    tail = *(node_data->tail[node_index]);
                    free_head = node_data->free[node_index];
                    node_data->free[node_index] = head;
                    tail->next = free_head;
                }

                /* Set the tail */
                node_data->tail[node_index] = node_ptr;
                node->next = NULL;

                break;
            }
            node_ptr = &(node->next);
            node = node->next;
        }

        /* Copy data from the location at tail to rhs and rhsold */
        from_node = *(node_data->tail[node_index]);
        to_node = &(node_data->rhs[node_index]);
        EVTnode_copy(ckt, node_index, from_node, &to_node);
        to_node = &(node_data->rhsold[node_index]);
        EVTnode_copy(ckt, node_index, from_node, &to_node);

    } /* end for number modified */

    /* Update/compact the modified list */
    for(i = 0, j = 0; i < num_modified; i++) {
        node_index = node_data->modified_index[i];
        /* If nothing after last_step, remove this index from the modified list */
        if((*(node_data->last_step[node_index]))->next == NULL) {
            node_data->modified[node_index] = MIF_FALSE;
            (node_data->num_modified)--;
        }
        /* else, keep the index */
        else {
            node_data->modified_index[j] = node_data->modified_index[i];
            j++;
        }
    }

} /* EVTbackup_node_data */



/*
EVTbackup_state_data()

Reset the state structure data.
*/


static void EVTbackup_state_data(
    CKTcircuit  *ckt,           /* the main circuit structure */
    double      new_time)       /* the time to backup to */
{
    int         i;
    int         j;

    int         num_modified;
    int         inst_index;

    Evt_State_Data_t    *state_data;

    Evt_State_t         **state_ptr;
    Evt_State_t         *state;
    Evt_State_t         *head;
    Evt_State_t         *tail;
    Evt_State_t         *free_head;

    /* Get pointers for quick access */
    state_data = ckt->evt->data.state;

    /* Loop through list of indexes modified since last accepted timepoint */
    num_modified = state_data->num_modified;
    for(i = 0; i < num_modified; i++) {

        /* Get the inst index */
        inst_index = state_data->modified_index[i];

        /* Scan data for this inst from last_step to determine new setting */
        /* for tail, and splice later data into the free list */
        state_ptr = state_data->last_step[inst_index];
        state = *state_ptr;
        for (;;) {
            if((state->next == NULL) || (state->next->step > new_time)) {

                /* Splice rest of list, if any, into free list */
                head = state->next;
                if(head) {
                    tail = *(state_data->tail[inst_index]);
                    free_head = state_data->free[inst_index];
                    state_data->free[inst_index] = head;
                    tail->next = free_head;
                }

                /* Set the tail */
                state_data->tail[inst_index] = state_ptr;
                state->next = NULL;

                break;
            }
            state_ptr = &(state->next);
            state = state->next;
        }
    } /* end for number modified */

    /* Update/compact the modified list */
    for(i = 0, j = 0; i < num_modified; i++) {
        inst_index = state_data->modified_index[i];
        /* If nothing after last_step, remove this index from the modified list */
        if((*(state_data->last_step[inst_index]))->next == NULL) {
            state_data->modified[inst_index] = MIF_FALSE;
            (state_data->num_modified)--;
        }
        /* else, keep the index */
        else {
            state_data->modified_index[j] = state_data->modified_index[i];
            j++;
        }
    }

} /* EVTbackup_state_data */



/*
EVTbackup_msg_data()

Backup the message data.
*/


static void EVTbackup_msg_data(
    CKTcircuit  *ckt,           /* the main circuit structure */
    double      new_time)       /* the time to backup to */
{
    int         i;
    int         j;

    int         num_modified;
    int         port_index;

    Evt_Msg_Data_t    *msg_data;

    Evt_Msg_t         **msg_ptr;
    Evt_Msg_t         *msg;
    Evt_Msg_t         *head;
    Evt_Msg_t         *tail;
    Evt_Msg_t         *free_head;

    /* Get pointers for quick access */
    msg_data = ckt->evt->data.msg;

    /* Loop through list of indexes modified since last accepted timepoint */
    num_modified = msg_data->num_modified;
    for(i = 0; i < num_modified; i++) {

        /* Get the port index */
        port_index = msg_data->modified_index[i];

        /* Scan data for this port from last_step to determine new setting */
        /* for tail, and splice later data into the free list */
        msg_ptr = msg_data->last_step[port_index];
        msg = *msg_ptr;
        for (;;) {
            if((msg->next == NULL) || (msg->next->step > new_time)) {

                /* Splice rest of list, if any, into free list */
                head = msg->next;
                if(head) {
                    tail = *(msg_data->tail[port_index]);
                    free_head = msg_data->free[port_index];
                    msg_data->free[port_index] = head;
                    tail->next = free_head;
                }

                /* Set the tail */
                msg_data->tail[port_index] = msg_ptr;
                msg->next = NULL;

                break;
            }
            msg_ptr = &(msg->next);
            msg = msg->next;
        }

    } /* end for number modified */

    /* Update/compact the modified list */
    for(i = 0, j = 0; i < num_modified; i++) {
        port_index = msg_data->modified_index[i];
        /* If nothing after last_step, remove this index from the modified list */
        if((*(msg_data->last_step[port_index]))->next == NULL) {
            msg_data->modified[port_index] = MIF_FALSE;
            (msg_data->num_modified)--;
        }
        /* else, keep the index */
        else {
            msg_data->modified_index[j] = msg_data->modified_index[i];
            j++;
        }
    }
}



/*
EVTbackup_inst_queue()

Backup data in inst queue.
*/


static void EVTbackup_inst_queue(
    CKTcircuit  *ckt,              /* the main circuit structure */
    double      new_time)          /* the time to backup to */
{

    int         i;
    int         j;

    int         num_modified;
    int         num_pending;
    int         inst_index;

    Evt_Inst_Queue_t    *inst_queue;

    Evt_Inst_Event_t    **inst_ptr;
    Evt_Inst_Event_t    *inst;

    double              next_time;
    double              event_time;


    /* Get pointers for quick access */
    inst_queue = &(ckt->evt->queue.inst);

    /* Loop through list of indexes modified since last accepted timepoint */
    /* and remove events with posted time > new_time */
    num_modified = inst_queue->num_modified;
    for(i = 0; i < num_modified; i++) {

        /* Get the inst index */
        inst_index = inst_queue->modified_index[i];

        /* Scan forward from last_step and cut out data with posted time */
        /* > new_time and add it to the free list */

        inst_ptr = inst_queue->last_step[inst_index];
        inst = *inst_ptr;

        while(inst) {
            if(inst->posted_time > new_time) {
                *inst_ptr = inst->next;
                inst->next = inst_queue->free[inst_index];
                inst_queue->free[inst_index] = inst;
                inst = *inst_ptr;
            }
            else {
                inst_ptr = &(inst->next);
                inst = *inst_ptr;
            }
        }

        /* Scan forward from last_step and set current to first */
        /* event with event_time > new_time */

        inst_ptr = inst_queue->last_step[inst_index];
        inst = *inst_ptr;

        while(inst) {
            if(inst->event_time > new_time)
                break;
            inst_ptr = &((*inst_ptr)->next);
            inst = *inst_ptr;
        }
        inst_queue->current[inst_index] = inst_ptr;
    }

    /* Add set of items modified to set of items pending before updating the */
    /* pending list because things may have been pulled from the pending list */
    for(i = 0; i < num_modified; i++) {
        j = inst_queue->modified_index[i];
        if(! inst_queue->pending[j]) {
            inst_queue->pending[j] = MIF_TRUE;
            inst_queue->pending_index[(inst_queue->num_pending)++] = j;
        }
    }

    /* Update the pending list and the next time by seeing if there */
    /* is anything at the location pointed to by current */
    next_time = 1e30;
    num_pending = inst_queue->num_pending;
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

    /* Update the modified list by looking for events that were processed
     * or queued in the current timestep.
     */

    for(i = 0, j = 0; i < num_modified; i++) {

        inst_index = inst_queue->modified_index[i];
        inst = *(inst_queue->last_step[inst_index]);

        if (inst_queue->current[inst_index] ==
            inst_queue->last_step[inst_index]) {
            /* Nothing now removed from the queue,
             * but it may have been modified by an addition.
             */

            while (inst) {
                if (inst->posted_time > inst_queue->last_time)
                    break;
                inst = inst->next;
            }
        }

        if(! inst) {
            inst_queue->modified[inst_index] = MIF_FALSE;
            (inst_queue->num_modified)--;
        }
        else {
            inst_queue->modified_index[j] = inst_queue->modified_index[i];
            j++;
        }
    }
}





/*
EVTbackup_output_queue()

Backup data in output queue.
*/



static void EVTbackup_output_queue(
    CKTcircuit  *ckt,               /* the main circuit structure */
    double      new_time)           /* the time to backup to */
{

    int         i;
    int         j;

    int         num_modified;
    int         num_pending;

   int         output_index;

    Evt_Output_Queue_t    *output_queue;

    Evt_Output_Event_t    **output_ptr, **free_list;
    Evt_Output_Event_t    *output;

    double              next_time;
    double              event_time;


    /* Get pointers for quick access */
    output_queue = &(ckt->evt->queue.output);

    /* Loop through list of indexes modified since last accepted timepoint */
    /* and remove events with posted time > new_time */
    num_modified = output_queue->num_modified;
    for(i = 0; i < num_modified; i++) {

        /* Get the output index */
        output_index = output_queue->modified_index[i];

        /* Scan forward from last_step and cut out data with posted time */
        /* > new_time and add it to the free list */
        /* Also, unremove anything with removed time > new_time */

        output_ptr = output_queue->last_step[output_index];
        output = *output_ptr;
        free_list = output_queue->free_list[output_index];

        while(output) {
            if(output->posted_time > new_time) {
                *output_ptr = output->next;
                output->next = *free_list;
                *free_list = output;
                output = *output_ptr;
            }
            else {
                if(output->removed && (output->removed_time > new_time))
                    output->removed = MIF_FALSE;
                output_ptr = &(output->next);
                output = *output_ptr;
            }
        }

        /* Scan forward from last_step and set current to first */
        /* event with event_time > new_time */

        output_ptr = output_queue->last_step[output_index];
        output = *output_ptr;

        while(output) {
            if(output->event_time > new_time)
                break;
            output_ptr = &((*output_ptr)->next);
            output = *output_ptr;
        }
        output_queue->current[output_index] = output_ptr;
    }

    /* Add set of items modified to set of items pending before updating the */
    /* pending list because things may have been pulled from the pending list */
    for(i = 0; i < num_modified; i++) {
        j = output_queue->modified_index[i];
        if(! output_queue->pending[j]) {
            output_queue->pending[j] = MIF_TRUE;
            output_queue->pending_index[(output_queue->num_pending)++] = j;
        }
    }

    /* Update the pending list and the next time by seeing if there */
    /* is anything at the location pointed to by current */
    next_time = 1e30;
    num_pending = output_queue->num_pending;
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

    /* Update the modified list by looking for events that were processed
     * or queued in the current timestep.
     */

    for(i = 0, j = 0; i < num_modified; i++) {

        output_index = output_queue->modified_index[i];
        output = *(output_queue->last_step[output_index]);

        if (output_queue->current[output_index] ==
            output_queue->last_step[output_index]) {
            /* Nothing now removed from the queue,
             * but it may have been modified by an addition.
	     */

            while(output) {
                if(output->posted_time > output_queue->last_time)
                    break;
                output = output->next;
            }
        }

        if(! output) {
            output_queue->modified[output_index] = MIF_FALSE;
            (output_queue->num_modified)--;
        }
        else {
            output_queue->modified_index[j] = output_queue->modified_index[i];
            j++;
        }
    }
}
