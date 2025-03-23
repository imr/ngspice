/*============================================================================
FILE    EVTaccept.c

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

    This file contains a function called at the end of a
    successful (accepted) analog timepoint.  It saves pointers
    to the states of the queues and data at this accepted time.

INTERFACES

    void EVTaccept(CKTcircuit *ckt, double time)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/*=== INCLUDE FILES ===*/

#include "ngspice/config.h"
#include <stdio.h>
#include "ngspice/cktdefs.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"
#include "ngspice/evtproto.h"



/*
EVTaccept()

This function is called at the end of a successful (accepted)
analog timepoint.  It saves pointers to the states of the
queues and data at this accepted time.
*/



void EVTaccept(
    CKTcircuit *ckt,    /* main circuit struct */
    double     time)    /* time at which analog soln was accepted */
{

    int         i;
    int         index;
    int         num_modified;

    Evt_Inst_Queue_t    *inst_queue;
    Evt_Output_Queue_t  *output_queue;
    Evt_Node_Info_t    **node_table;
    Evt_Node_Data_t     *node_data;
    Evt_State_Data_t    *state_data;
    Evt_Msg_Data_t      *msg_data;


    /* Exit if no event instances */
    if(ckt->evt->counts.num_insts == 0)
        return;

    /* Get often used pointers */
    inst_queue = &(ckt->evt->queue.inst);
    output_queue = &(ckt->evt->queue.output);
    node_table = ckt->evt->info.node_table;
    node_data = ckt->evt->data.node;
    state_data = ckt->evt->data.state;
    msg_data = ckt->evt->data.msg;


    /* Process the inst queue */
    num_modified = inst_queue->num_modified;
    /* Loop through list of items modified since last time */
    for(i = 0; i < num_modified; i++) {
        Evt_Inst_Event_t *stale, *next;

        /* Get the index of the inst modified */
        index = inst_queue->modified_index[i];
        /* Reset the modified flag */
        inst_queue->modified[index] = MIF_FALSE;
        /* Move stale entries to the free list. */
        next = inst_queue->head[index];
        while (next) {
            if (next->event_time >= time ||
                &next->next == inst_queue->current[index]) {
                break;
            }
            stale = next;
            next = next->next;
            stale->next = inst_queue->free[index];
            inst_queue->free[index] = stale;
        }
        inst_queue->head[index] = next;
        if (!next)
            inst_queue->current[index] = &inst_queue->head[index];
        /* Update last_step for this index */
        inst_queue->last_step[index] = inst_queue->current[index];
    }
    /* Record the new last_time and reset number modified to zero */
    inst_queue->last_time = time;
    inst_queue->num_modified = 0;


    /* Process the output queue */
    num_modified = output_queue->num_modified;
    /* Loop through list of items modified since last time */
    for(i = 0; i < num_modified; i++) {
        Evt_Output_Event_t *stale, *next, **free_list;

        /* Get the index of the output modified */
        index = output_queue->modified_index[i];
        /* Reset the modified flag */
        output_queue->modified[index] = MIF_FALSE;
        /* Move stale entries to the free list. */
        free_list = output_queue->free_list[index];
        next = output_queue->head[index];
        while (next) {
            if (next->event_time >= time ||
                &next->next == output_queue->current[index]) {
                break;
            }
            stale = next;
            next = next->next;
            stale->next = *free_list;
            *free_list = stale;
        }
        output_queue->head[index] = next;
        if (!next)
            output_queue->current[index] = &output_queue->head[index];
        /* Update last_step for this index */
        output_queue->last_step[index] = output_queue->current[index];
    }
    /* Record the new last_time and reset number modified to zero */
    output_queue->last_time = time;
    output_queue->num_modified = 0;


    /* Process the node data */
    num_modified = node_data->num_modified;
    /* Loop through list of items modified since last time */
    for(i = 0; i < num_modified; i++) {
        Evt_Node_t      *this;
        Evt_Node_Info_t *node_info;
        Evt_Node_Cb_t   *cb, **cbpp;
        int              udn_index;

        /* Get the index of the node modified */
        index = node_data->modified_index[i];
        /* Reset the modified flag */
        node_data->modified[index] = MIF_FALSE;

        /* Call any value-change functions registered for this node. */

        node_info = node_table[index];
        udn_index = node_info->udn_index;

        cbpp = &node_info->cbs;
        for (;;) {
            Mif_Value_t val;

            cb = *cbpp;
            if (cb == NULL)
                break;

            for (this = *node_data->last_step[index];
                 this;
                 this = this->next) {
                switch (cb->type) {
                case Evt_Cbt_Raw:
                    val.pvalue = this->node_value;
                    break;
                case Evt_Cbt_Plot:
                    g_evt_udn_info[udn_index]->plot_val(this->node_value,
                                                        (char *)cb->member,
                                                        &val.rvalue);
                    break;
                }

                if ((*cb->fn)(this->step, &val, cb->ctx, !this->next)) {
                    /* Remove callback from chain. */

                    *cbpp = cb->next;
                    txfree(cb);
                    break;
                }
            }
            if (this == NULL)   // Normal loop exit.
                cbpp = &cb->next;
        }

        /* Optionally store node values for later examination.
         * The test of CKTtime here is copied from dctran.c.
         * CKTinitTime is from the tstart parameter of the "tran"
         * command or card.
         */

        if (node_info->save && ckt->CKTtime >= ckt->CKTinitTime &&
            (ckt->CKTtime > 0 || !(ckt->CKTmode & MODEUIC))) {
            /* Update last_step for this index */
            node_data->last_step[index] = node_data->tail[index];
        } else {
            Evt_Node_t *keep;

            /* If not recording history, discard all but the last item.
             * It may be needed to restore the previous state on backup.
             */
            keep = *(node_data->tail[index]);
            *(node_data->tail[index]) = node_data->free[index];
            node_data->free[index] = node_data->head[index];
            node_data->head[index] = keep;
            node_data->last_step[index] = node_data->tail[index] =
                &node_data->head[index];
        }
    }
    /* Reset number modified to zero */
    node_data->num_modified = 0;


    /* Process the state data */
    num_modified = state_data->num_modified;
    /* Loop through list of items modified since last time */
    for(i = 0; i < num_modified; i++) {
        Evt_State_t         *state;

        /* Get the index of the state modified */
        index = state_data->modified_index[i];
        /* Reset the modified flag */
        state_data->modified[index] = MIF_FALSE;
        /* Get the last saved state for this instance. */
        state = *(state_data->tail[index]);
        /* Dump everything older on the instance-specific free list,
         * recreating the setup after the initial calls to cm_event_alloc().
         */
        if (!state)
            continue;
        if (state->prev) {
            state->prev->next = state_data->free[index];
            state_data->free[index] = state_data->head[index];
        }
        state_data->head[index] = state;
        state_data->last_step[index] = state_data->tail[index] =
            &state_data->head[index];
    }
    /* Reset number modified to zero */
    state_data->num_modified = 0;


    /* Process the msg data */
    num_modified = msg_data->num_modified;
    /* Loop through list of items modified since last time */
    for(i = 0; i < num_modified; i++) {
        /* Get the index of the msg modified */
        index = msg_data->modified_index[i];
        /* Update last_step for this index */
        msg_data->last_step[index] = msg_data->tail[index];
        /* Reset the modified flag */
        msg_data->modified[index] = MIF_FALSE;
    }
    /* Reset number modified to zero */
    msg_data->num_modified = 0;

} /* EVTaccept */


/* Functions to set-up and cancel value-changed callbacks. */

Mif_Boolean_t EVTnew_value_call(const char         *node,
                                Evt_New_Value_Cb_t  fn,
                                Evt_Node_Cb_Type_t  type,
                                void               *ctx)
{
    struct node_parse  result;
    int                index;

    Evt_Ckt_Data_t    *evt;
    CKTcircuit        *ckt;
    Evt_Node_Info_t   *node_info;
    Evt_Node_Cb_t     *cb;

    index = Evt_Parse_Node(node, &result);
    if (index < 0)
        return MIF_FALSE;
    ckt = g_mif_info.ckt;
    evt = ckt->evt;
    node_info = evt->info.node_table[index];
    cb = tmalloc(sizeof *cb);
    cb->next = node_info->cbs;
    node_info->cbs = cb;
    cb->fn = fn;
    cb->type = type;
    cb->member = copy(result.member);
    cb->ctx = ctx;
    txfree(result.node);
    return MIF_TRUE;
}

void EVTcancel_value_call(const char         *node,
                          Evt_New_Value_Cb_t  fn,
                          void               *ctx)
{
    Evt_Ckt_Data_t    *evt;
    CKTcircuit        *ckt;
    Evt_Node_Info_t  **node_table, *node_info;
    Evt_Node_Cb_t    **cbpp, *cb;
    int                i, num_nodes;

    ckt = g_mif_info.ckt;
    if (!ckt)
        return;
    evt = ckt->evt;
    if (!evt)
        return;

    /* Look for node name in the event-driven node list */

    node_table = evt->info.node_table;
    num_nodes = evt->counts.num_nodes;

    for (i = 0; i < num_nodes; i++) {
        if (cieq(node, node_table[i]->name))
            break;
    }
    if (i >= num_nodes)
        return;

    node_info = node_table[i];
    cbpp = &node_info->cbs;
    cb = node_info->cbs;
    while (cb) {
        if (cb->fn == fn && cb->ctx == ctx) {
            *cbpp = cb->next;
            tfree(cb);
        } else {
            cbpp = &cb->next;
        }
        cb = *cbpp;
    }
}
