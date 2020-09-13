/*============================================================================
FILE    EVTiter.c

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

    This file contains function EVTiter which iterates through
    event-driven outputs and instances until the outputs no longer change.


INTERFACES

    int EVTiter(CKTcircuit *ckt)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
//#include "misc.h"
#include "ngspice/cktdefs.h"
//#include "util.h"
#include "ngspice/sperror.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/enh.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"


/*
EVTiter

This function iterates through event-driven outputs and instances
until the outputs no longer change.  The general algorithm used
is:

Do:

    Scan list of changed outputs
        Put items on the node queue 'to_eval' list for
        each changed output.

    Scan list of changed nodes
        Resolve nodes with multiple outputs posted.
        Create inverted state for nodes with attached
        inverted inputs.
        Put items on the instance queue 'to_call' list
        for each changed node.
        If transient analysis, put state of the node
        into the node data structure.

    Scan instance to_call list
        Call EVTload for each instance on list.

While there are changed outputs

*/


int EVTiter(
    CKTcircuit *ckt)    /* the circuit structure */
{

    int         i;
    int         num_changed;

    int         num_to_eval;
    int         num_to_call;

    int         output_index;
    /*    int         output_subindex;*/
    int         inst_index;
    int         node_index;
    int         port_index;

    int         num_outputs;
    int         udn_index;

    int         passes;

    Evt_Ckt_Data_t  *evt;

    Evt_Output_Queue_t  *output_queue;
    Evt_Node_Queue_t    *node_queue;
    Evt_Inst_Queue_t    *inst_queue;

    Evt_Output_Info_t   **output_table;
    Evt_Node_Info_t     **node_table;
    Evt_Port_Info_t     **port_table;

    Evt_Inst_Index_t    *inst_list;

    Evt_Node_Data_t     *node_data;

    Evt_Node_t          *rhs;
    Evt_Node_t          *rhsold;

    Evt_Node_t          *node;

    Mif_Boolean_t       equal;

    char                *err_msg;


    /* Get temporary pointers for fast access */
    evt = ckt->evt;

    output_queue = &(evt->queue.output);
    node_queue = &(evt->queue.node);
    inst_queue = &(evt->queue.inst);

    output_table = evt->info.output_table;
    node_table = evt->info.node_table;
    port_table = evt->info.port_table;

    node_data = evt->data.node;
    rhs = node_data->rhs;
    rhsold = node_data->rhsold;


    /* Loop until no more output change, or too many passes through loop */
    for(passes = 0; passes < evt->limits.max_event_passes; passes++) {


        /* Create list of nodes to evaluate from list of changed outputs */
        num_changed = output_queue->num_changed;
        for(i = 0; i < num_changed; i++) {

            /* Get index of node that output is connected to */
            output_index = output_queue->changed_index[i];
            node_index = output_table[output_index]->node_index;

            /* If not already on list of nodes to evaluate, add it */
            if(! node_queue->to_eval[node_index]) {
                node_queue->to_eval[node_index] = MIF_TRUE;
                node_queue->to_eval_index[(node_queue->num_to_eval)++]
                        = node_index;
            }

            /* Reset the changed flag on the output queue */
            output_queue->changed[output_index] = MIF_FALSE;

        }
        output_queue->num_changed = 0;



        /* Evaluate nodes and for any which have changed, enter */
        /* the instances that receive inputs from them on the list */
        /* of instances to call */

        num_to_eval = node_queue->num_to_eval;
        for(i = 0; i < num_to_eval; i++) {

            /* Get the node index, udn index and number of outputs */
            node_index = node_queue->to_eval_index[i];
            udn_index = node_table[node_index]->udn_index;
            num_outputs = node_table[node_index]->num_outputs;

            /* Resolve the node value if multiple outputs on it */
            /* and test if new node value is different than old value */
            if(num_outputs > 1) {
                g_evt_udn_info[udn_index]->resolve
                        (num_outputs,
                        rhs[node_index].output_value,
                        rhs[node_index].node_value);
                g_evt_udn_info[udn_index]->compare
                        (rhs[node_index].node_value,
                        rhsold[node_index].node_value,
                        &equal);
                if(! equal) {
                    g_evt_udn_info[udn_index]->copy
                            (rhs[node_index].node_value,
                            rhsold[node_index].node_value);
                }
            }
            /* Else, load function has already determined that they were */
            /* not equal */
            else
                equal = MIF_FALSE;

            /* If not equal, make inverted copy in rhsold if */
            /* needed, and place indexes of instances with inputs connected */
            /* to the node in the to_call list of inst queue */
            if(! equal) {
                if(node_table[node_index]->invert) {
                    g_evt_udn_info[udn_index]->copy
                            (rhsold[node_index].node_value,
                            rhsold[node_index].inverted_value);
                    g_evt_udn_info[udn_index]->invert
                            (rhsold[node_index].inverted_value);
                }
                inst_list = node_table[node_index]->inst_list;
                while(inst_list) {
                    inst_index = inst_list->index;
                    if(! inst_queue->to_call[inst_index]) {
                        inst_queue->to_call[inst_index] = MIF_TRUE;
                        inst_queue->to_call_index[(inst_queue->num_to_call)++]
                                = inst_index;
                    }
                    inst_list = inst_list->next;
                } /* end while instances with inputs on node */
            } /* end if not equal */

            /* If transient analysis mode */
            /* Save the node data onto the node results list and mark */
            /* that it has been modified, even if the */
            /* resolved node value has not changed */
            if(g_mif_info.circuit.anal_type == MIF_TRAN) {

                node = *(node_data->tail[node_index]);
                node_data->tail[node_index] = &(node->next);
                EVTnode_copy(ckt, node_index, &(rhsold[node_index]), &(node->next));
                node->next->step = g_mif_info.circuit.evt_step;

                if(! node_data->modified[node_index]) {
                    node_data->modified[node_index] = MIF_TRUE;
                    node_data->modified_index[(node_data->num_modified)++] = node_index;
                }
            }

            /* Reset the to_eval flag on the node queue */
            node_queue->to_eval[node_index] = MIF_FALSE;

        } /* end for number of nodes to evaluate */
        node_queue->num_to_eval = 0;



        /* Call the instances with inputs on nodes that have changed */
        num_to_call = inst_queue->num_to_call;
        for(i = 0; i < num_to_call; i++) {
            inst_index = inst_queue->to_call_index[i];
            inst_queue->to_call[inst_index] = MIF_FALSE;
            EVTload(ckt, inst_index);
        }
        inst_queue->num_to_call = 0;


        /* Record statistics */
        if(g_mif_info.circuit.anal_type == MIF_DC)
            (ckt->evt->data.statistics->op_event_passes)++;


        /* If no outputs changed, iteration is over, so return with success! */
        if(output_queue->num_changed == 0)
            return(0);

    } /* end for */


    /* Too many passes through loop, report problems and exit with error */

    err_msg = TMALLOC(char, 10000);
    for(i = 0; i < output_queue->num_changed; i++) {
        output_index = output_queue->changed_index[i];
        port_index = output_table[output_index]->port_index;
        sprintf(err_msg, "\n    Instance: %s\n    Connection: %s\n    Port: %d",
                port_table[port_index]->inst_name,
                port_table[port_index]->conn_name,
                port_table[port_index]->port_num);
        ENHreport_conv_prob(ENH_EVENT_NODE,
                            port_table[port_index]->node_name,
                            err_msg);
    }
    FREE(err_msg);

    SPfrontEnd->IFerrorf (ERR_WARNING,
        "Too many iteration passes in event-driven circuits");
    return(E_ITERLIM);

}
