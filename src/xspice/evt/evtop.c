/*============================================================================
FILE    EVTop.c

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

    This file contains function EVTop which is used to perform an
    operating point analysis in place of CKTop when there are
    event-driven instances in the circuit.  It alternates between doing
    event-driven iterations with EVTiter and doing analog iterations with
    NIiter/CKTop until no more event-driven outputs change.

INTERFACES

    EVTop()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
//#include "util.h"
#include "ngspice/sperror.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/enh.h"
#include "ngspice/evtproto.h"
#include "ngspice/evtudn.h"


static void EVTnode_compare(
    CKTcircuit    *ckt,
    int           node_index,
    Evt_Node_t    *node1,
    Evt_Node_t    *node2,
    Mif_Boolean_t *equal);



/*
EVTop

This function is used to perform an operating point analysis
in place of CKTop when there are event-driven instances in the
circuit.  It alternates between doing event-driven iterations
with EVTiter and doing analog iterations with NIiter/CKTop
until no more event-driven outputs change.
*/


int EVTop(
    CKTcircuit     *ckt,          /* The circuit structure */
    long           firstmode,     /* The SPICE 3C1 CKTop() firstmode parameter */
    long           continuemode,  /* The SPICE 3C1 CKTop() continuemode paramter */
    int            max_iter,      /* The SPICE 3C1 CKTop() max iteration parameter */
    Mif_Boolean_t  first_call)    /* Is this the first time through? */
{

    int         i;
    int         num_insts;
    int         converged;
    int         output_index;
    int         port_index;

    char        *err_msg;

    Mif_Boolean_t       firstime;

    Evt_Inst_Queue_t    *inst_queue;
    Evt_Output_Queue_t  *output_queue;

    Evt_Output_Info_t   **output_table;
    Evt_Port_Info_t     **port_table;


    /* get data to local storage for fast access */
    num_insts = ckt->evt->counts.num_insts;
    inst_queue = &(ckt->evt->queue.inst);

    /* Initialize to_call entries in event inst queue */
    /* to force calling all event/hybrid instance the first */
    /* time through */

    if(first_call) {
        for(i = 0; i < num_insts; i++) {
            inst_queue->to_call[i] = MIF_TRUE;
            inst_queue->to_call_index[i] = i;
        }
        inst_queue->num_to_call = num_insts;
    }


    /* Alternate between event-driven and analog solutions until */
    /* there are no changed event-driven outputs */

    firstime = MIF_TRUE;
    for(;;) {

        /* Call EVTiter to establish initial outputs from */
        /* event/hybrid instances with states (e.g. flip-flops) */

        ckt->CKTmode = firstmode;
        converged = EVTiter(ckt);
        if(converged != 0)
            return(converged);

        /* Now do analog solution for current state of hybrid outputs */

        /* If first analog solution, call CKTop */
        if(firstime) {
            firstime = MIF_FALSE;
            converged = CKTop(ckt,
                              firstmode,
                              continuemode,
                              max_iter);
            if(converged != 0)
                return(converged);
        }
        /* Otherwise attempt to converge with mode = continuemode */
        else {
            ckt->CKTmode = continuemode;
            converged = NIiter(ckt,max_iter);
            if(converged != 0) {
                converged = CKTop(ckt,
                              firstmode,
                              continuemode,
                              max_iter);
                if(converged != 0)
                    return(converged);
            }
        }

        /* Call all hybrids to allow new event outputs to be posted */
        EVTcall_hybrids(ckt);

        /* Increment count of successful alternations */
        (ckt->evt->data.statistics->op_alternations)++;

        /* If .option card specified not to alternate solutions, exit */
        /* immediately with this first pass solution */
        if(! ckt->evt->options.op_alternate)
            return(0);

        /* If no hybrid instances produced different event outputs, */
        /* alternation is completed, so exit */
        if(ckt->evt->queue.output.num_changed == 0)
            return(0);

        /* If too many alternations, exit with error */
        if(ckt->evt->data.statistics->op_alternations >=
                ckt->evt->limits.max_op_alternations) {

            SPfrontEnd->IFerrorf (ERR_WARNING,
                "Too many analog/event-driven solution alternations");

            err_msg = TMALLOC(char, 10000);
            output_queue = &(ckt->evt->queue.output);
            output_table = ckt->evt->info.output_table;
            port_table = ckt->evt->info.port_table;

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

            return(E_ITERLIM);
        }

    } /* end forever */
}



/*
EVTop_save

Save result from operating point iteration into the node data area.
*/

void EVTop_save(
    CKTcircuit    *ckt,  /* The circuit structure */
    Mif_Boolean_t op,    /* True if from a DCOP analysis, false if TRANOP, etc. */
    double        step)
{

    int         i;
    int         num_nodes;

    Mif_Boolean_t       equal;

    Evt_Node_Data_t     *node_data;

    Evt_Node_t          *rhsold;
    Evt_Node_t          **head;
    Evt_Node_t          **here;

    /*	char buff[128];*/


    /* Get pointers for fast access */
    node_data = ckt->evt->data.node;
    rhsold = node_data->rhsold;
    head = node_data->head;

    /* For number of event nodes, copy rhsold to node data */
    /* and set the op member if appropriate */
    num_nodes = ckt->evt->counts.num_nodes;

    for(i = 0; i < num_nodes; i++) 
	{
        /* if head is null, just copy there */
        if(head[i] == NULL) 
		{
            EVTnode_copy(ckt, i, &(rhsold[i]), &(head[i]));
			
            head[i]->op = op;
            head[i]->step = step;
			
        }
        /* Otherwise, add to the end of the list */
        else 
		{
            /* Locate end of list */
            here = &(head[i]);
            for(;;) 
			{
                if((*here)->next)
                    here = &((*here)->next);
                else
                    break;
            }
            /* Compare entry at end of list to rhsold */
			
            EVTnode_compare(ckt, i, &(rhsold[i]), *here, &equal);
			
            /* If new value in rhsold is different, add it to the list */
            if(!equal) 
			{
                here = &((*here)->next);
                EVTnode_copy(ckt, i, &(rhsold[i]), here);
                (*here)->op = op;
                (*here)->step = step;
            }
        } /* end else add to end of list */
    }  /* end for number of nodes */

}



/* ************************************************************ */


/*
EVTnode_compare

This function compares the resolved values of the old and
new states on a node.  The actual comparison is done by
calling the appropriate user-defined node compare function.
*/


static void EVTnode_compare(
    CKTcircuit    *ckt,        /* The circuit structure */
    int           node_index,  /* The index for the node in question */
    Evt_Node_t    *node1,      /* The first value */
    Evt_Node_t    *node2,      /* The second value */
    Mif_Boolean_t *equal)      /* The computed result */
{

    Evt_Node_Data_t     *node_data;
    Evt_Node_Info_t     **node_table;

    int                 udn_index;


    /* Get data for fast access */
    node_data = ckt->evt->data.node;
    node_table = ckt->evt->info.node_table;
    udn_index = node_table[node_index]->udn_index;


    /* Do compare based on changes in resolved node value only */
    g_evt_udn_info[udn_index]->compare (
            node1->node_value,
            node2->node_value,
            equal);
}
