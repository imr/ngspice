/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTdestroy(ckt)
     * this is a driver program to iterate through all the various
     * destroy functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
static int evt_dest(Evt_Ckt_Data_t *evt);
#endif

int
CKTdestroy(CKTcircuit *ckt)
{
    int i;
    CKTnode *node;
    CKTnode *nnode;


#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
         if(ckt->CKTrhsOp) FREE(ckt->CKTrhsOp);
         if(ckt->CKTsenRhs) FREE(ckt->CKTsenRhs);
         if(ckt->CKTseniRhs) FREE(ckt->CKTseniRhs);
         SENdestroy(ckt->CKTsenInfo);
    }
#endif

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVdestroy && ckt->CKThead[i] ) {
            DEVices[i]->DEVdestroy (&(ckt->CKThead[i]));
        }
    }
    for(i=0;i<=ckt->CKTmaxOrder+1;i++){
        FREE(ckt->CKTstates[i]);
    }
    if(ckt->CKTmatrix) {
        SMPdestroy(ckt->CKTmatrix);
        ckt->CKTmatrix = NULL;
    }
    FREE(ckt->CKTbreaks);
    for(node = ckt->CKTnodes; node; ) {
        nnode = node->next;
        FREE(node);
        node = nnode;
    }
    ckt->CKTnodes = NULL;
    ckt->CKTlastNode = NULL;

    FREE(ckt->CKTrhs);
    FREE(ckt->CKTrhsOld);
    FREE(ckt->CKTrhsSpare);
    FREE(ckt->CKTirhs);
    FREE(ckt->CKTirhsOld);
    FREE(ckt->CKTirhsSpare);

    FREE(ckt->CKTstat->STATdevNum);
    FREE(ckt->CKTstat);
    FREE(ckt->CKThead);

#ifdef XSPICE
    evt_dest(ckt->evt);
    FREE(ckt->enh);
    FREE(ckt->evt);
#endif

    nghash_free(ckt->DEVnameHash, NULL, NULL);
    FREE(ckt);
    return(OK);
}

#ifdef XSPICE
static int
evt_dest(Evt_Ckt_Data_t *evt)
{
    int i;

    /* Get temporary pointers for fast access */
    Evt_Output_Queue_t  *output_queue;
    Evt_Node_Queue_t    *node_queue;
    Evt_Inst_Queue_t    *inst_queue;

    Evt_Output_Info_t   **output_table;
    Evt_Node_Info_t     **node_table;
    Evt_Port_Info_t     **port_table;

    Evt_State_Data_t    *state_data;
    Evt_State_t         *statenext, *state;

    Evt_Node_Data_t     *node_data;
    Evt_Node_t          *rhs, *rhsnext;

    Evt_Node_t          *node, *nodenext;

    Evt_Msg_Data_t      *msg_data;
    Evt_Msg_t           *msg, *msgnext;

    Evt_Inst_Event_t    *here;
    Evt_Inst_Event_t    *next;

    Evt_Output_Event_t  *outhere;
    Evt_Output_Event_t  *outnext;

    /* Exit immediately if no event-driven instances in circuit */
    if (evt->counts.num_insts == 0)
        return(OK);

    output_queue = &(evt->queue.output);
    node_queue = &(evt->queue.node);
    inst_queue = &(evt->queue.inst);

    output_table = evt->info.output_table;
    node_table = evt->info.node_table;
    port_table = evt->info.port_table;

    node_data = evt->data.node;

    state_data = evt->data.state;
    msg_data = evt->data.msg;

    /* instance queue */
    for (i = 0; i < evt->counts.num_insts; i++) {
        here = inst_queue->head[i];
        while (here) {
            next = here->next;
            tfree(here);
            here = next;
        }
    }
    tfree(inst_queue->head);
    tfree(inst_queue->current);
    tfree(inst_queue->last_step);
    tfree(inst_queue->free);

    tfree(inst_queue->modified_index);
    tfree(inst_queue->modified);
    tfree(inst_queue->pending_index);
    tfree(inst_queue->pending);
    tfree(inst_queue->to_call_index);
    tfree(inst_queue->to_call);

    /* node queue */

    tfree(node_queue->to_eval_index);
    tfree(node_queue->to_eval);
    tfree(node_queue->changed_index);
    tfree(node_queue->changed);

    /* output queue */
    for (i = 0; i < evt->counts.num_outputs; i++) {
        outhere = output_queue->head[i];
        while (outhere) {
            outnext = outhere->next;
            tfree(outhere);
            outhere = outnext;
        }
    }
    tfree(output_queue->head);
    tfree(output_queue->current);
    tfree(output_queue->last_step);
    tfree(output_queue->free);

    tfree(output_queue->modified_index);
    tfree(output_queue->modified);
    tfree(output_queue->pending_index);
    tfree(output_queue->pending);
    tfree(output_queue->changed_index);
    tfree(output_queue->changed);

    /* state data */
    /* only if digital states are there */
    if (state_data) {
        for (i = 0; i < evt->counts.num_insts; i++) {
            state = state_data->head[i];
            while (state) {
                statenext = state->next;
                tfree(state->block);
                tfree(state);
                state = statenext;
            }
        }

        tfree(state_data->head);
        tfree(state_data->tail);
        tfree(state_data->last_step);
        tfree(state_data->free);

        tfree(state_data->modified);
        tfree(state_data->modified_index);
        tfree(state_data->total_size);
        /* FIXME: more on desc ? */
        tfree(state_data->desc);
    }

    /* node data */
    /* only if digital nodes are there */
    if (node_data) {
        for (i = 0; i < evt->counts.num_nodes; i++) {
            node = node_data->head[i];
            while (node) {
                nodenext = node->next;
                tfree(node);
                node = nodenext;
            }
        }
        tfree(node_data->head);
        tfree(node_data->tail);
        tfree(node_data->last_step);
        tfree(node_data->free);

        tfree(node_data->modified);
        tfree(node_data->modified_index);

        for (i = 0; i < evt->counts.num_nodes; i++) {
            rhs = &(node_data->rhs[i]);
            while (rhs) {
                rhsnext = rhs->next;
                tfree(rhs->inverted_value);
                tfree(rhs->output_value);
                tfree(rhs->node_value);
                rhs = rhsnext;
            }
        }
        tfree(node_data->rhs);
        for (i = 0; i < evt->counts.num_nodes; i++) {
            rhs = &(node_data->rhsold[i]);
            while (rhs) {
                rhsnext = rhs->next;
                tfree(rhs->inverted_value);
                tfree(rhs->output_value);
                tfree(rhs->node_value);
                rhs = rhsnext;
            }
        }
        tfree(node_data->rhsold);
        tfree(node_data->total_load);
    }

    /* msg data */

    if (msg_data) {
        for (i = 0; i < evt->counts.num_ports; i++) {
            msg = msg_data->head[i];
            while (msg) {
                msgnext = msg->next;
                tfree(msg);
                msg = msgnext;
            }
        }

        tfree(msg_data->head);
        tfree(msg_data->tail);
        tfree(msg_data->last_step);
        tfree(msg_data->free);

        tfree(msg_data->modified);
        tfree(msg_data->modified_index);
    }

    /* still to care for :
     *   output_table
     *   node_table
     *   port_table
     */

    return(OK);
}
#endif
