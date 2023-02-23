/*============================================================================
FILE    EVTinit.c

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

    This file contains function EVTinit which allocates and initializes
    evt structure elements after the number of instances, nodes, etc.
    have been determined in parsing during INPpas2.  EVTinit also checks
    to be sure no nodes have been used for both analog and event-driven
    algorithms simultaneously.

INTERFACES

    int EVTinit(CKTcircuit *ckt)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
#include  "ngspice/cktdefs.h"
//#include  "util.h"
#include  "ngspice/sperror.h"

#include  "ngspice/evt.h"
#include  "ngspice/evtproto.h"


static int EVTcount_hybrids(CKTcircuit *ckt);
static int EVTinit_info(CKTcircuit *ckt);
static int EVTinit_queue(CKTcircuit *ckt);
static int EVTinit_limits(CKTcircuit *ckt);



/* Allocation macro with built-in check for out-of-memory */
/* Adapted from SPICE 3C1 code in CKTsetup.c */
#define CKALLOC(var,size,type) \
    if(size) { \
        if((var = TMALLOC(type, size)) == NULL) \
            return(E_NOMEM); \
    }


/*
EVTinit

Allocate and initialize additional evt structure elements now that
we can determine the number of instances, nodes, etc.

Also check to be sure that no nodes have been used in both event-driven
and analog domains.

In this version, we also report an error if there are no hybrids in the
circuit.  This restriction may be removed in the future to allow the
simulator to be used with digital only circuits...
*/


int EVTinit(
    CKTcircuit  *ckt)   /* the circuit structure */
{

    int  err;           /* SPICE error return code   0 = OK */

    /* Exit immediately if there are no event-driven instances */
    /* but don't complain */
    if(ckt->evt->counts.num_insts == 0)
        return(OK);

    /* Count the number of hybrids and hybrid outputs */
    err = EVTcount_hybrids(ckt);
    if(err)
        return(err);

    /* Create info table arrays */
    err = EVTinit_info(ckt);
    if(err)
        return(err);

    /* Setup queues */
    err = EVTinit_queue(ckt);
    if(err)
        return(err);

    /* Initialize limits */
    err = EVTinit_limits(ckt);
    if(err)
        return(err);

    /* Note:  Options were initialized in CKTinit so that INPpas2  */
    /* could set values according to .options cards in deck.  The  */
    /* structure 'jobs' will be setup immediately prior to each    */
    /* simulation job.  The results data structure is also         */
    /* allocated immediately prior to each simulation job.         */

    /* Return */
    return(OK);
}


/*
EVTcount_hybrids

Count the number of hybrids and the number of outputs on all hybrids.
*/

static int EVTcount_hybrids(
    CKTcircuit *ckt)          /* The circuit structure */
{

    int i;
    int j;

    int num_hybrids;
    int num_hybrid_outputs;
    int num_conn;
    int num_port;

    MIFinstance         *fast;

    Evt_Inst_Info_t     *inst;


    /* Count number of hybrids and hybrid outputs in the inst list */
    /* created during parsing.  Note: other counts */
    /* are created during parsing, but these were */
    /* too difficult to do until now... */
    num_hybrids = 0;
    num_hybrid_outputs = 0;
    inst = ckt->evt->info.inst_list;
    while(inst) {
        fast = inst->inst_ptr;
        if(fast->analog && fast->event_driven) {
            num_hybrids++;
            num_conn = fast->num_conn;
            for(i = 0; i < num_conn; i++) {
                if((! fast->conn[i]->is_null) && (fast->conn[i]->is_output)) {
                    num_port = fast->conn[i]->size;
                    for(j = 0; j < num_port; j++)
                        if(! fast->conn[i]->port[j]->is_null)
                            num_hybrid_outputs++;
                }
            }
        }
        inst = inst->next;
    }
    ckt->evt->counts.num_hybrids = num_hybrids;
    ckt->evt->counts.num_hybrid_outputs = num_hybrid_outputs;

    return(OK);
}


/*
EVTinit_info

This function creates the ``info'' pointer tables used in the
event-driven circuit representation.  These arrays allow faster
access to data associated with instances, nodes, ports, and
outputs than could be provided by having to scan the linked-list
representations created during parsing.
*/


static int EVTinit_info(
    CKTcircuit *ckt)       /* the circuit structure */
{

    int i;
    int j;

    int num_insts;
    int num_nodes;
    int num_ports;
    int num_outputs;

    Evt_Inst_Info_t     *inst;
    Evt_Node_Info_t     *node;
    Evt_Port_Info_t     *port;
    Evt_Output_Info_t   *output;

    Evt_Inst_Info_t     **inst_table = NULL;
    Evt_Node_Info_t     **node_table = NULL;
    Evt_Port_Info_t     **port_table = NULL;
    Evt_Output_Info_t   **output_table = NULL;

    int                 *hybrid_index = NULL;

    int num_hybrids;


    /* Allocate and initialize table of inst pointers */
    num_insts = ckt->evt->counts.num_insts;
    CKALLOC(inst_table, num_insts, Evt_Inst_Info_t *)
    inst = ckt->evt->info.inst_list;
    for(i = 0; i < num_insts; i++) {
        inst_table[i] = inst;
        inst = inst->next;
    }
    ckt->evt->info.inst_table = inst_table;

    /* Allocate and initialize table of node pointers */
    num_nodes = ckt->evt->counts.num_nodes;
    CKALLOC(node_table, num_nodes, Evt_Node_Info_t *)
    node = ckt->evt->info.node_list;
    for(i = 0; i < num_nodes; i++) {
        node_table[i] = node;
        node = node->next;
    }
    ckt->evt->info.node_table = node_table;

    /* Allocate and initialize table of port pointers */
    num_ports = ckt->evt->counts.num_ports;
    CKALLOC(port_table, num_ports, Evt_Port_Info_t *)
    port = ckt->evt->info.port_list;
    for(i = 0; i < num_ports; i++) {
        port_table[i] = port;
        port = port->next;
    }
    ckt->evt->info.port_table = port_table;

    /* Allocate and initialize table of output pointers */
    num_outputs = ckt->evt->counts.num_outputs;
    CKALLOC(output_table, num_outputs, Evt_Output_Info_t *)
    output = ckt->evt->info.output_list;
    for(i = 0; i < num_outputs; i++) {
        output_table[i] = output;
        output = output->next;
    }
    ckt->evt->info.output_table = output_table;


    /* Allocate and create table of indexes into inst_table for hybrids */
    num_hybrids = ckt->evt->counts.num_hybrids;
    CKALLOC(hybrid_index, num_hybrids, int)
    for(i = 0, j = 0; i < num_insts; i++) {
        if(inst_table[i]->inst_ptr->analog)
            hybrid_index[j++] = i;
    }
    ckt->evt->info.hybrid_index = hybrid_index;


    /* Return */
    return(OK);
}



/*
EVTinit_queue

This function prepares the event-driven queues for simulation.
*/


static int EVTinit_queue(
    CKTcircuit *ckt)       /* the circuit structure */
{

    int num_insts;
    int num_nodes;
    int num_outputs;

    Evt_Inst_Queue_t    *inst_queue;
    Evt_Node_Queue_t    *node_queue;
    Evt_Output_Queue_t  *output_queue;


    /* Allocate elements in the inst queue */

    num_insts = ckt->evt->counts.num_insts;
    inst_queue = &(ckt->evt->queue.inst);

    CKALLOC(inst_queue->head, num_insts, Evt_Inst_Event_t *)
    CKALLOC(inst_queue->current, num_insts, Evt_Inst_Event_t **)
    CKALLOC(inst_queue->last_step, num_insts, Evt_Inst_Event_t **)
    CKALLOC(inst_queue->free, num_insts, Evt_Inst_Event_t *)
    CKALLOC(inst_queue->modified_index, num_insts, int)
    CKALLOC(inst_queue->modified, num_insts, Mif_Boolean_t)
    CKALLOC(inst_queue->pending_index, num_insts, int)
    CKALLOC(inst_queue->pending, num_insts, Mif_Boolean_t)
    CKALLOC(inst_queue->to_call_index, num_insts, int)
    CKALLOC(inst_queue->to_call, num_insts, Mif_Boolean_t)


    /* Allocate elements in the node queue */

    num_nodes = ckt->evt->counts.num_nodes;
    node_queue = &(ckt->evt->queue.node);

    CKALLOC(node_queue->to_eval_index, num_nodes, int)
    CKALLOC(node_queue->to_eval, num_nodes, Mif_Boolean_t)
    CKALLOC(node_queue->changed_index, num_nodes, int)
    CKALLOC(node_queue->changed, num_nodes, Mif_Boolean_t)


    /* Allocate elements in the output queue */

    num_outputs = ckt->evt->counts.num_outputs;
    output_queue = &(ckt->evt->queue.output);

    CKALLOC(output_queue->head, num_outputs, Evt_Output_Event_t *)
    CKALLOC(output_queue->current, num_outputs, Evt_Output_Event_t **)
    CKALLOC(output_queue->last_step, num_outputs, Evt_Output_Event_t **)
    CKALLOC(output_queue->free_list, num_outputs, Evt_Output_Event_t **)
    CKALLOC(output_queue->modified_index, num_outputs, int)
    CKALLOC(output_queue->modified, num_outputs, Mif_Boolean_t)
    CKALLOC(output_queue->pending_index, num_outputs, int)
    CKALLOC(output_queue->pending, num_outputs, Mif_Boolean_t)
    CKALLOC(output_queue->changed_index, num_outputs, int)
    CKALLOC(output_queue->changed, num_outputs, Mif_Boolean_t)

    /* Return */
    return(OK);
}




/*
EVTinit_limits

This function initializes the iteration limits applicable to the
event-driven algorithm.
*/


static int EVTinit_limits(
    CKTcircuit *ckt)         /* the circuit structure */
{

    /* Set maximum number of event load calls within a single event iteration */
    /* to the number of event outputs.  This should allow for the        */
    /* maximum possible number of events that can trickle through any    */
    /* circuit that does not contain loops.                              */

    ckt->evt->limits.max_event_passes = ckt->evt->counts.num_outputs + 1;


    /* Set maximum number of alternations between analog and event-driven */
    /* iterations to the number of event outputs on hybrids.              */

    ckt->evt->limits.max_op_alternations = ckt->evt->counts.num_hybrid_outputs + 1;


    /* Return */
    return(OK);
}
