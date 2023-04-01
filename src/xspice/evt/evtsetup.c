/*============================================================================
FILE    EVTsetup.c

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

    This file contains function EVTsetup which clears/allocates the
    event-driven queues and data structures immediately prior to a new
    analysis.  In addition, it places entries in the job list so that
    results data from multiple analysis can be retrieved similar to
    SPICE3C1 saving multiple 'plots'.

INTERFACES

    int EVTsetup(CKTcircuit *ckt)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
//#include "misc.h"

#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
//#include "util.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"
#include "ngspice/mifproto.h"
#include "ngspice/evtproto.h"




static int EVTsetup_queues(CKTcircuit *ckt);
static int EVTsetup_data(CKTcircuit *ckt);
static int EVTsetup_jobs(CKTcircuit *ckt);
static int EVTsetup_load_ptrs(CKTcircuit *ckt);

int EVTsetup_plot(CKTcircuit* ckt, char* plottypename);
int EVTswitch_plot(CKTcircuit* ckt, const char* plottypename);


/* Allocation macros with built-in check for out-of-memory */
/* Adapted from SPICE 3C1 code in CKTsetup.c */

#define CKALLOC(var,size,type) \
    if(size) { \
        if((var = TMALLOC(type, size)) == NULL) \
            return(E_NOMEM); \
    }

#define CKREALLOC(var,size,type) \
    if((size) == 1) { \
        if((var = TMALLOC(type, size)) == NULL) \
            return(E_NOMEM); \
    } else if((size) > 1) { \
        if((var = TREALLOC(type, (var), size)) == NULL) \
            return(E_NOMEM); \
    }


/*
EVTsetup

This function clears/allocates the event-driven queues and data structures
immediately prior to a new analysis.  In addition, it places entries
in the job list so that results data from multiple analysis can be
retrieved similar to SPICE3C1 saving multiple 'plots'.
*/


int EVTsetup(
    CKTcircuit *ckt)   /* The circuit structure */
{

    int err;


    /* Exit immediately if no event-driven instances in circuit */
    if(ckt->evt->counts.num_insts == 0)
        return(OK);

    /* Clear the inst, node, and output queues, and initialize the to_call */
    /* elements in the instance queue to call all event-driven instances */
    err = EVTsetup_queues(ckt);
    if(err)
        return(err);

    /* Allocate and initialize the node, state, message, and statistics data */
    err = EVTsetup_data(ckt);
    if(err)
        return(err);

    /* Set the job pointers to the allocated results, states, messages, */
    /* and statistics so that data will be accessable after run */
    err = EVTsetup_jobs(ckt);
    if(err)
        return(err);

    /* Setup the pointers in the MIFinstance structure for inputs, outputs, */
    /* and total loads */
    err = EVTsetup_load_ptrs(ckt);
    if(err)
        return(err);

    /* Initialize additional event data */
    g_mif_info.circuit.evt_step = 0.0;

    /* Return OK */
    return(OK);
}


int EVTunsetup(
    CKTcircuit* ckt)   /* The circuit structure */
{
    int err;

    /* Exit immediately if no event-driven instances in circuit */
    if (ckt->evt->counts.num_insts == 0)
        return(OK);

    /* Clear the inst, node, and output queues, and initialize the to_call */
    /* elements in the instance queue to call all event-driven instances */
    err = EVTsetup_queues(ckt);
    if (err)
        return(err);

    /* Initialize additional event data */
    g_mif_info.circuit.evt_step = 0.0;

    /* Return OK */
    return(OK);
}




/*
EVTsetup_queues

This function clears the event-driven queues in preparation for
a new simulation.
*/


static int EVTsetup_queues(
    CKTcircuit *ckt)         /* The circuit structure */
{

    int i;
    int num_insts;
    int num_nodes;
    int num_outputs;

    Evt_Inst_Queue_t    *inst_queue;
    Evt_Node_Queue_t    *node_queue;
    Evt_Output_Queue_t  *output_queue;

    Evt_Inst_Event_t    *inst_event;
    Evt_Output_Event_t  *output_event;
    void                *ptr;

    /* ************************ */
    /* Clear the instance queue */
    /* ************************ */

    num_insts = ckt->evt->counts.num_insts;
    inst_queue = &(ckt->evt->queue.inst);

    for(i = 0; i < num_insts; i++) {
        inst_event = inst_queue->head[i];
        while(inst_event) {
            ptr = inst_event;
            inst_event = inst_event->next;
            FREE(ptr);
        }
        inst_event = inst_queue->free[i];
        while(inst_event) {
            ptr = inst_event;
            inst_event = inst_event->next;
            FREE(ptr);
        }
        inst_queue->head[i] = NULL;
        inst_queue->current[i] = &(inst_queue->head[i]);
        inst_queue->last_step[i] = &(inst_queue->head[i]);
        inst_queue->free[i] = NULL;
    }

    inst_queue->next_time = 0.0;
    inst_queue->last_time = 0.0;

    inst_queue->num_modified = 0;
    inst_queue->num_pending = 0;
    inst_queue->num_to_call = 0;

    for(i = 0; i < num_insts; i++) {
        inst_queue->modified[i] = MIF_FALSE;
        inst_queue->pending[i] = MIF_FALSE;
        inst_queue->to_call[i] = MIF_FALSE;
    }


    /* ******************** */
    /* Clear the node queue */
    /* ******************** */

    num_nodes = ckt->evt->counts.num_nodes;
    node_queue = &(ckt->evt->queue.node);

    node_queue->num_changed = 0;
    node_queue->num_to_eval = 0;

    for(i = 0; i < num_nodes; i++) {
        node_queue->changed[i] = MIF_FALSE;
        node_queue->to_eval[i] = MIF_FALSE;
    }

    /* ********************** */
    /* Clear the output queue */
    /* ********************** */

    num_outputs = ckt->evt->counts.num_outputs;
    output_queue = &(ckt->evt->queue.output);

    for(i = 0; i < num_outputs; i++) {
        output_event = output_queue->head[i];
        while(output_event) {
            ptr = output_event;
            output_event = output_event->next;
            FREE(ptr);
        }
        output_queue->head[i] = NULL;
        output_queue->current[i] = &(output_queue->head[i]);
        output_queue->last_step[i] = &(output_queue->head[i]);
    }

    output_queue->next_time = 0.0;
    output_queue->last_time = 0.0;

    output_queue->num_modified = 0;
    output_queue->num_pending = 0;
    output_queue->num_changed = 0;

    if (num_outputs > 0) {
        for (i = 0; i < num_outputs; i++) {
            output_queue->modified[i] = MIF_FALSE;
            output_queue->pending[i] = MIF_FALSE;
            output_queue->changed[i] = MIF_FALSE;
        }

        if (output_queue->free_list[0]) {
            Evt_purge_free_outputs();
        } else {
            Evt_Output_Info_t *output_info;
            Evt_Node_Info_t   *node;

            /* ********************************************************* *
             * On first call for this circuit, set the free-list pointer
             * for each output queue.
             * ********************************************************* */

            output_info = ckt->evt->info.output_list;
            for (i = 0; i < num_outputs; i++) {
                node =  ckt->evt->info.node_table[output_info->node_index];
                output_queue->free_list[i] =
                    &g_evt_udn_info[node->udn_index]->free_list;
                output_info = output_info->next;
            }
        }
    }
    return OK;
}

void Evt_purge_free_outputs(void)
{
    Evt_Output_Event_t  *output_event, *next;
    int                  i;

    for (i = 0; i < g_evt_num_udn_types; ++i) {
        output_event = g_evt_udn_info[i]->free_list;
        g_evt_udn_info[i]->free_list = NULL;
        while (output_event) {
            next = output_event->next;
            tfree(output_event->value);
            tfree(output_event);
            output_event = next;
        }
    }
}


/*
EVTsetup_data

This function sets up the event-driven node, state, and
message data runtime structures in preparation for
a new simulation.
*/


static int EVTsetup_data(
    CKTcircuit *ckt)       /* The circuit structure */
{

    Evt_Data_t  *data;

    int         i;
    int         j;

    int         num_insts;
    int         num_ports;
    int         num_nodes;

    int         udn_index;
    int         num_outputs;

    Mif_Boolean_t  invert;

    Evt_Node_Data_t     *node_data;
    Evt_State_Data_t    *state_data;
    Evt_Msg_Data_t      *msg_data;
    /*    Evt_Statistic_t     *statistics_data;*/

    Evt_Node_t          *rhs;
    Evt_Node_t          *rhsold;

    Evt_Node_Info_t     *node_info;


    /* Allocate main substructures of data */
    /* Note that we don't free any old structures */
    /* since they are pointed to by jobs and need */
    /* to be maintained so that results from multiple */
    /* jobs are kept around like SPICE does */

    data = &(ckt->evt->data);
    CKALLOC(data->node, 1, Evt_Node_Data_t)
    CKALLOC(data->state, 1, Evt_State_Data_t)
    CKALLOC(data->msg, 1, Evt_Msg_Data_t)
    CKALLOC(data->statistics, 1, Evt_Statistic_t)

    /* Allocate node data */

    num_nodes = ckt->evt->counts.num_nodes;
    node_data = data->node;

    CKALLOC(node_data->head, num_nodes, Evt_Node_t *)
    CKALLOC(node_data->tail, num_nodes, Evt_Node_t **)
    CKALLOC(node_data->last_step, num_nodes, Evt_Node_t **)
    CKALLOC(node_data->free, num_nodes, Evt_Node_t *)
    CKALLOC(node_data->modified_index, num_nodes, int)
    CKALLOC(node_data->modified, num_nodes, Mif_Boolean_t)
    CKALLOC(node_data->rhs, num_nodes, Evt_Node_t)
    CKALLOC(node_data->rhsold, num_nodes, Evt_Node_t)
    CKALLOC(node_data->total_load, num_nodes, double)

    /* Initialize the node data */

    for(i = 0; i < num_nodes; i++) {
        node_data->tail[i] = &(node_data->head[i]);
        node_data->last_step[i] = &(node_data->head[i]);
    }

    for(i = 0; i < num_nodes; i++) {

        /* Get pointers to rhs & rhsold, the user-defined node type index, */
        /* the number of outputs on the node and the invert flag */
        rhs = &(node_data->rhs[i]);
        rhsold = &(node_data->rhsold[i]);
        node_info = ckt->evt->info.node_table[i];
        udn_index = node_info->udn_index;
        num_outputs = node_info->num_outputs;
        invert = node_info->invert;

        /* Initialize the elements within rhs and rhsold */
        rhs->step = 0.0;
        rhsold->step = 0.0;
        if(num_outputs > 1) {
            CKALLOC(rhs->output_value, num_outputs, void *)
            CKALLOC(rhsold->output_value, num_outputs, void *)
            for(j = 0; j < num_outputs; j++) {
                g_evt_udn_info[udn_index]->create (&(rhs->output_value[j]));
                g_evt_udn_info[udn_index]->initialize (rhs->output_value[j]);
                g_evt_udn_info[udn_index]->create (&(rhsold->output_value[j]));
                g_evt_udn_info[udn_index]->initialize (rhsold->output_value[j]);
            }
        }
        g_evt_udn_info[udn_index]->create (&(rhs->node_value));
        g_evt_udn_info[udn_index]->initialize (rhs->node_value);
        g_evt_udn_info[udn_index]->create (&(rhsold->node_value));
        g_evt_udn_info[udn_index]->initialize (rhsold->node_value);
        if(invert) {
            g_evt_udn_info[udn_index]->create (&(rhs->inverted_value));
            g_evt_udn_info[udn_index]->initialize (rhs->inverted_value);
            g_evt_udn_info[udn_index]->create (&(rhsold->inverted_value));
            g_evt_udn_info[udn_index]->initialize (rhsold->inverted_value);
        }

        /* Initialize the total load value to zero */
        node_data->total_load[i] = 0.0;
    }


    /* Allocate and initialize state data */

    num_insts = ckt->evt->counts.num_insts;
    state_data = data->state;

    CKALLOC(state_data->head, num_insts, Evt_State_t *)
    CKALLOC(state_data->tail, num_insts, Evt_State_t **)
    CKALLOC(state_data->last_step, num_insts, Evt_State_t **)
    CKALLOC(state_data->free, num_insts, Evt_State_t *)
    CKALLOC(state_data->modified_index, num_insts, int)
    CKALLOC(state_data->modified, num_insts, Mif_Boolean_t)
    CKALLOC(state_data->total_size, num_insts, int)
    CKALLOC(state_data->desc, num_insts, Evt_State_Desc_t *)

    for(i = 0; i < num_insts; i++) {
        state_data->tail[i] = &(state_data->head[i]);
        state_data->last_step[i] = &(state_data->head[i]);
    }


    /* Allocate and initialize msg data */

    num_ports = ckt->evt->counts.num_ports;
    msg_data = data->msg;

    CKALLOC(msg_data->head, num_ports, Evt_Msg_t *)
    CKALLOC(msg_data->tail, num_ports, Evt_Msg_t **)
    CKALLOC(msg_data->last_step, num_ports, Evt_Msg_t **)
    CKALLOC(msg_data->free, num_ports, Evt_Msg_t *)
    CKALLOC(msg_data->modified_index, num_ports, int)
    CKALLOC(msg_data->modified, num_ports, Mif_Boolean_t)

    for(i = 0; i < num_ports; i++) {
        msg_data->tail[i] = &(msg_data->head[i]);
        msg_data->last_step[i] = &(msg_data->head[i]);
    }

    /* Don't need to initialize statistics since they were */
    /* calloc'ed above */

    return(OK);
}




/*
EVTsetup_jobs

This function prepares the jobs data for a new simulation.
*/


static int EVTsetup_jobs(
    CKTcircuit *ckt)       /* The circuit structure */
{

    int  i;
    int  num_jobs;

    Evt_Job_t  *jobs;
    Evt_Data_t *data;


    jobs = &(ckt->evt->jobs);
    data = &(ckt->evt->data);

    /* Increment the number of jobs */
    num_jobs = ++(jobs->num_jobs);

    /* Allocate/reallocate necessary pointers */
    CKREALLOC(jobs->job_name, num_jobs, char *)
    CKREALLOC(jobs->job_plot, num_jobs, char *)
    CKREALLOC(jobs->node_data, num_jobs, Evt_Node_Data_t *)
    CKREALLOC(jobs->state_data, num_jobs, Evt_State_Data_t *)
    CKREALLOC(jobs->msg_data, num_jobs, Evt_Msg_Data_t *)
    CKREALLOC(jobs->statistics, num_jobs, Evt_Statistic_t *)

    /* Fill in the pointers, etc. for this new job */
    i = num_jobs - 1;
    jobs->job_name[i] = MIFcopy(ckt->CKTcurJob->JOBname);
    jobs->job_plot[i] = NULL; /* fill in later */
    jobs->node_data[i] = data->node;
    jobs->state_data[i] = data->state;
    jobs->msg_data[i] = data->msg;
    jobs->statistics[i] = data->statistics;

    return(OK);
}



/*
EVTsetup_load_ptrs

This function setups up the required data in the MIFinstance
structure of event-driven and hybrid instances.
*/


static int EVTsetup_load_ptrs(
    CKTcircuit *ckt)            /* The circuit structure */
{

    int         i;
    int         j;
    int         k;

    int         num_insts;
    int         num_conn;
    int         num_port;
    int         num_outputs;

    int         node_index;
    int         output_subindex;

    MIFinstance         *fast;

    Mif_Conn_Data_t     *conn;

    Mif_Port_Type_t     type;
    Mif_Port_Data_t     *port;

    Evt_Node_Data_t     *node_data;


    /* This function setups up the required data in the MIFinstance */
    /* structure of event-driven and hybrid instances */

    /* Loop through all event-driven and hybrid instances */
    num_insts = ckt->evt->counts.num_insts;
    for(i = 0; i < num_insts; i++) {

        /* Get the MIFinstance pointer */
        fast = ckt->evt->info.inst_table[i]->inst_ptr;

        /* Reset init flag, required when any run is called a second time */
        fast->initialized = FALSE;

        /* Loop through all connections */
        num_conn = fast->num_conn;
        for(j = 0; j < num_conn; j++) {

            /* Skip if connection is null */
            if(fast->conn[j]->is_null)
                continue;

            conn = fast->conn[j];

            /* Loop through all ports */
            num_port = conn->size;
            for(k = 0; k < num_port; k++) {

                /* Get port data pointer for quick access */
                port = conn->port[k];

                if(port->is_null)
                    continue;

                /* Skip if port is not digital or user-defined type */
                type = port->type;
                if((type != MIF_DIGITAL) && (type != MIF_USER_DEFINED))
                    continue;

                /* Set input.pvalue to point to rhsold.node_value or to */
                /* rhsold.inverted_value as appropriate */
                node_index = port->evt_data.node_index;
                node_data  = ckt->evt->data.node;
                if(conn->is_input) {
                    if(port->invert) {
                        port->input.pvalue = node_data->rhsold[node_index].
                                             inverted_value;
                    }
                    else {
                        port->input.pvalue = node_data->rhsold[node_index].
                                             node_value;
                    }
                }

                /* Set output.pvalue to point to rhs.node_value or rhs.output_value[i] */
                /* where i is given by the output_subindex in output info */
                /* depending on whether more than one output is connected to the node. */
                /* Note that this is only for the DCOP analysis.  During a transient */
                /* analysis, new structures will be created and the pointers will */
                /* be set by EVTload */
                if(conn->is_output) {
                    num_outputs = ckt->evt->info.node_table[node_index]->num_outputs;
                    if(num_outputs <= 1) {
                        port->output.pvalue = node_data->rhs[node_index].
                                              node_value;
                    }
                    else {
                        output_subindex = port->evt_data.output_subindex;
                        port->output.pvalue = node_data->rhs[node_index].
                                              output_value[output_subindex];
                    }
                }

            } /* end for number of ports */
        } /* end for number of connections */
    } /* end for number of insts */

    return(OK);
}

/* get the analog plot name and store it into the current event job */
int EVTsetup_plot(CKTcircuit* ckt, char *plotname) {
    if (ckt->evt->counts.num_insts == 0)
        return(OK);
    
    Evt_Job_t* jobs = &(ckt->evt->jobs);
    if (jobs) {
        jobs->job_plot[jobs->num_jobs - 1] = copy(plotname);
        jobs->cur_job = jobs->num_jobs - 1;
        return OK;
    }
    return 1;
}

/* If command 'setplot' is called, we switch to the corresponding event data.
   Their pointers have been stored in the jobs structure. The circuit must
   be still available!
*/
int EVTswitch_plot(CKTcircuit* ckt, const char* plottypename) {
    int i;
    bool found = FALSE;

    Evt_Job_t* jobs;
    Evt_Data_t* data;

    if (!ckt)
        return (E_NOTFOUND);

    if (ckt->evt->counts.num_insts == 0)
        return(E_NOTFOUND);

    jobs = &(ckt->evt->jobs);
    data = &(ckt->evt->data);

    if (jobs) {
        /* check for the job with current plot type name , e.g. tran2 */
        for (i = 0; i < jobs->num_jobs; i++) {
            if (jobs->job_plot[i] && eq(jobs->job_plot[i], plottypename)) {
                found = TRUE;
                jobs->cur_job = i;
                break;
            }
        }
        if (found) {
            data->node = jobs->node_data[i];
            data->state = jobs->state_data[i];
            data->msg = jobs->msg_data[i];
            data->statistics = jobs->statistics[i];
            return OK;
        }
    }
    return 1;
}
