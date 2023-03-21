/*============================================================================
FILE    EVTload.c

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

    This file contains function EVTload which is used to call a
    specified event-driven or hybrid code model during an event-driven
    iteration.  The 'CALL_TYPE' is set to 'EVENT_DRIVEN' when the
    model is called from this function.

INTERFACES

    int EVTload(CKTcircuit *ckt, int inst_index)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
//#include "util.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/mifproto.h"
#include "ngspice/evtproto.h"
#include "ngspice/cmproto.h"


static void EVTcreate_state(
    CKTcircuit  *ckt,
    int         inst_index);

static void EVTadd_msg(
    CKTcircuit  *ckt,
    int         port_index,
    char        *msg_text);

static Evt_Output_Event_t *EVTget_output_event(
    CKTcircuit      *ckt,
    Mif_Port_Data_t *port);

static void EVTprocess_output(
    CKTcircuit          *ckt,
    Mif_Port_Data_t     *port);

/*
EVTload

This function calls the code model function for the specified
instance with CALL_TYPE set to EVENT_DRIVEN.  Event outputs,
messages, etc. are processed on return from the code model.
Analog outputs, partials, etc. should not be computed by the
code model when the call type is event-driven and are
ignored.
*/

int EVTload(
    CKTcircuit *ckt,        /* The circuit structure */
    int        inst_index)  /* The instance to call code model for */
{

    int                 i;
    int                 j;

    int                 num_conn;
    int                 num_port;
    int                 mod_type;

    Mif_Conn_Data_t     *conn;
    Mif_Port_Data_t     *port;
    Evt_Node_Data_t     *node_data;
    MIFinstance         *inst;

    Mif_Private_t       cm_data;


    /* ***************************** */
    /* Prepare the code model inputs */
    /* ***************************** */

    /* Get pointer to instance data structure and other data */
    /* needed for fast access */
    inst = ckt->evt->info.inst_table[inst_index]->inst_ptr;
    node_data = ckt->evt->data.node;

    /* Setup circuit data in struct to be passed to code model function */

    if(inst->initialized)
        cm_data.circuit.init = MIF_FALSE;
    else
        cm_data.circuit.init = MIF_TRUE;

    cm_data.circuit.anal_init = MIF_FALSE;
    cm_data.circuit.anal_type = g_mif_info.circuit.anal_type;

    if(g_mif_info.circuit.anal_type == MIF_TRAN)
        cm_data.circuit.time = g_mif_info.circuit.evt_step;
    else
        cm_data.circuit.time = 0.0;

    cm_data.circuit.call_type = MIF_EVENT_DRIVEN;
    cm_data.circuit.temperature = ckt->CKTtemp - 273.15;

    /* Setup data needed by cm_... functions */

    g_mif_info.ckt = ckt;
    g_mif_info.instance = inst;
    g_mif_info.errmsg = "";
    g_mif_info.circuit.call_type = MIF_EVENT_DRIVEN;

    if(inst->initialized)
        g_mif_info.circuit.init = MIF_FALSE;
    else
        g_mif_info.circuit.init = MIF_TRUE;


    /* If after initialization and in transient analysis mode */
    /* create a new state for the instance */

    if((g_mif_info.circuit.anal_type == MIF_TRAN) && inst->initialized)
        EVTcreate_state(ckt, inst_index);


    /* Loop through all connections on the instance and setup */
    /* load, total_load, and msg on all ports, and changed flag */
    /* and output pointer on all outputs */

    num_conn = inst->num_conn;
    for(i = 0; i < num_conn; i++) {

        conn = inst->conn[i];

        /* if connection is null, continue to next */
        if(conn->is_null)
            continue;

        /* Loop through each port on the connection */
        num_port = conn->size;
        for(j = 0; j < num_port; j++) {

            port = conn->port[j];

            /* Skip if port is null */
            if(port->is_null)
                continue;

            /* If port type is Digital or User-Defined */
            if((port->type == MIF_DIGITAL) || (port->type == MIF_USER_DEFINED)) {

                /* Initialize the msg pointer on the port to NULL, */
                /* initialize the load value to zero, and get the total load */
                port->msg = NULL;
                port->load = 0.0;
                port->total_load = node_data->total_load[port->evt_data.node_index];

                /* If connection is an output and transient analysis,
                 * initialize changed to true and ensure an output location.
                 */
                if(conn->is_output) {
                    port->changed = MIF_TRUE;
                    if (g_mif_info.circuit.anal_type == MIF_TRAN) {
                        if (port->next_event == NULL) {
                            port->next_event = EVTget_output_event(ckt, port);
                        }
                        port->output.pvalue = port->next_event->value;
                    }
                }
            }
            else {
                /* Get the analog input value.  All we need to do is */
                /* set it to zero if mode is INITJCT.  Otherwise, value */
                /* should still be around from last successful analog call */
                if(ckt->CKTmode & MODEINITJCT)
                    port->input.rvalue = 0.0;
            }

        } /* end for number of ports */
    } /* end for number of connections */


    /* Prepare the structure to be passed to the code model */
    cm_data.num_conn = inst->num_conn;
    cm_data.conn = inst->conn;
    cm_data.num_param = inst->num_param;
    cm_data.param = inst->param;
    cm_data.num_inst_var = inst->num_inst_var;
    cm_data.inst_var = inst->inst_var;
    cm_data.callback = &(inst->callback);


    /* ******************* */
    /* Call the code model */
    /* ******************* */

    mod_type = MIFmodPtr(inst)->MIFmodType;
    DEVices[mod_type]->DEVpublic.cm_func (&cm_data);


    /* ****************************** */
    /* Process the code model outputs */
    /* ****************************** */

    /* Loop through all connections and ports and process the msgs */
    /* and event outputs */

    num_conn = inst->num_conn;
    for(i = 0; i < num_conn; i++) {

        conn = inst->conn[i];
        if(conn->is_null)
            continue;

        /* Loop through each port on the connection */
        num_port = conn->size;
        for(j = 0; j < num_port; j++) {

            port = conn->port[j];

            /* Skip if port is null */
            if(port->is_null)
                continue;

            /* Process the message if any */
            if(port->msg)
                EVTadd_msg(ckt, port->evt_data.port_index, port->msg);

            /* If this is the initialization pass, process the load factor */
            if(! inst->initialized) {
                node_data->total_load[port->evt_data.node_index] +=
                        port->load;
            }

            /* If connection is not an event output, continue to next port */
            if(! conn->is_output)
                continue;
            if((port->type != MIF_DIGITAL) && (port->type != MIF_USER_DEFINED))
                continue;

            /* If output changed, process it */

            if (port->changed)
                EVTprocess_output(ckt, port);

            /* And prevent erroneous models from overwriting it during */
            /* analog iterations */
            if(g_mif_info.circuit.anal_type == MIF_TRAN)
                port->output.pvalue = NULL;

        } /* end for number of ports */
    } /* end for number of connections */


    /* Record statistics */
    if(g_mif_info.circuit.anal_type == MIF_DC)
        (ckt->evt->data.statistics->op_load_calls)++;
    else if(g_mif_info.circuit.anal_type == MIF_TRAN)
        (ckt->evt->data.statistics->tran_load_calls)++;

    /* Mark that the instance has been called once */
    inst->initialized = MIF_TRUE;

    return(OK);
}



/*
EVTcreate_state

This function creates a new state storage area for a particular instance
during an event-driven simulation.  New states must be created so
that old states are saved and can be accessed by code models in the
future.  The new state is initialized to the previous state value.
*/


static void EVTcreate_state(
    CKTcircuit  *ckt,         /* The circuit structure */
    int         inst_index)   /* The instance to create state for */
{
    size_t              total_size;

    Evt_State_Data_t    *state_data;

    Evt_State_t         *new_state;
    Evt_State_t         *prev_state;

    /* Get variables for fast access */
    state_data = ckt->evt->data.state;

    /* Exit immediately if no states on this instance */
    if(state_data->desc[inst_index] == NULL)
        return;

    /* Get size of state block to be allocated */
    total_size = (size_t) state_data->total_size[inst_index];

    /* Allocate a new state for the instance */
    if(state_data->free[inst_index]) 
	{
        new_state = state_data->free[inst_index];
        state_data->free[inst_index] = new_state->next;
        new_state->next = NULL; // reusing dirty memory: next must be reset
    }
    else 
	{
		
        new_state = TMALLOC(Evt_State_t, 1);
        new_state->block = tmalloc(total_size);

    }

    /* Splice the new state into the state data linked list */
    /* and update the tail pointer */
    prev_state = *(state_data->tail[inst_index]);
    prev_state->next = new_state;
    new_state->prev = prev_state;
    state_data->tail[inst_index] = &(prev_state->next);

    /* Copy the old state to the new state and set the step */
    memcpy(new_state->block, prev_state->block, total_size);
    new_state->step = g_mif_info.circuit.evt_step;

    /* Mark that the state data on the instance has been modified */
    if(! state_data->modified[inst_index]) {
        state_data->modified[inst_index] = MIF_TRUE;
        state_data->modified_index[(state_data->num_modified)++] = inst_index;
    }
}


/*
EVTget_output_event

This function creates a new output event.
*/

static Evt_Output_Event_t *EVTget_output_event(
    CKTcircuit      *ckt,       /* The circuit structure */
    Mif_Port_Data_t *port)      /* The output port. */
{
    int                 udn_index;
    Evt_Node_Info_t     **node_table;
    Evt_Output_Queue_t  *output_queue;
    Evt_Output_Event_t  *event, **free_list;


    /* Check the output queue free list and use the structure */
    /* at the head of the list if non-null.  Otherwise, create a new one. */

    output_queue = &(ckt->evt->queue.output);
    free_list = output_queue->free_list[port->evt_data.output_index];
    if (*free_list) {
        event = *free_list;
        *free_list = event->next;
    } else {
        /* Create a new event */
        event = TMALLOC(Evt_Output_Event_t, 1);
        event->next = NULL;

        /* Initialize the value */
        node_table = ckt->evt->info.node_table;
        udn_index = node_table[port->evt_data.node_index]->udn_index;
        g_evt_udn_info[udn_index]->create (&(event->value));
    }
    return event;
}



/*
EVTadd_msg

This function records a message output by a code model into the
message results data structure.
*/


static void EVTadd_msg(
    CKTcircuit  *ckt,          /* The circuit structure */
    int         port_index,    /* The port to add message to */
    char        *msg_text)     /* The message text */
{

    Evt_Msg_Data_t      *msg_data;

    Evt_Msg_t           **msg_ptr;
    Evt_Msg_t           *msg;


    /* Get pointers for fast access */
    msg_data = ckt->evt->data.msg;
    msg_ptr = msg_data->tail[port_index];

    /* Set pointer to location at which to add, and update tail */
    if(*msg_ptr != NULL) {
        msg_ptr = &((*msg_ptr)->next);
        msg_data->tail[port_index] = msg_ptr;
    }

    /* Add a new entry in the list of messages for this port */
    if(msg_data->free[port_index]) {
        *msg_ptr = msg_data->free[port_index];
        msg_data->free[port_index] = msg_data->free[port_index]->next;
        if ((*msg_ptr)->text)
            tfree((*msg_ptr)->text);
    }
    else {
        *msg_ptr = TMALLOC(Evt_Msg_t, 1);
    }

    /* Fill in the values */
    msg = *msg_ptr;
    msg->next = NULL;
    if((ckt->CKTmode & MODEDCOP) == MODEDCOP)
        msg->op = MIF_TRUE;
    else
        msg->step = g_mif_info.circuit.evt_step;
    msg->text = MIFcopy(msg_text);

    /* Update the modified indexes */
    if(g_mif_info.circuit.anal_type == MIF_TRAN) {
        if(! msg_data->modified[port_index]) {
            msg_data->modified[port_index] = MIF_TRUE;
            msg_data->modified_index[(msg_data->num_modified)++] = port_index;
        }
    }

}


/* This is a code-model library function.  Placed here to use local
 * static functions.
 */

bool cm_schedule_output(unsigned int conn_index, unsigned int port_index,
                        double delay, void *vp)
{
    MIFinstance        *instance;
    Mif_Conn_Data_t    *conn;
    Mif_Port_Data_t    *port;
    Evt_Node_Info_t    *node_info;
    Evt_Output_Event_t *output_event;
    int                 udn_index;

    if (delay < 0 || g_mif_info.circuit.anal_type != MIF_TRAN)
        return FALSE;
    instance = g_mif_info.instance;
    if (conn_index >= (unsigned int)instance->num_conn)
        return FALSE;
    conn = instance->conn[conn_index];
    if (port_index >= (unsigned int)conn->size)
        return FALSE;
    port = conn->port[port_index];
    if (port->type != MIF_DIGITAL && port->type != MIF_USER_DEFINED)
        return FALSE;

    /* Get an output structure and copy the new value. */

    output_event = EVTget_output_event(g_mif_info.ckt, port);
    node_info =
        g_mif_info.ckt->evt->info.node_table[port->evt_data.node_index];
    udn_index = node_info->udn_index;
    g_evt_udn_info[node_info->udn_index]->copy(vp, output_event->value);

    /* Queue the output. */

    if (port->invert)
        g_evt_udn_info[udn_index]->invert(output_event->value);
    EVTqueue_output(g_mif_info.ckt, port->evt_data.output_index,
                    udn_index, output_event,
                    g_mif_info.circuit.evt_step,
                    g_mif_info.circuit.evt_step + delay);
    return TRUE;
}

/*
EVTprocess_output

This function processes an event-driven output produced by a code
model.  If transient analysis mode, the event is placed into the
output queue according to its (non-zero) delay.  If DC analysis,
the event is processed immediately.
*/


static void EVTprocess_output(
    CKTcircuit      *ckt,          /* The circuit structure */
    Mif_Port_Data_t *port)
{

    int                 num_outputs;
    int                 node_index;
    int                 udn_index;
    int                 output_index;
    int                 output_subindex;

    Evt_Output_Info_t   **output_table;
    Evt_Node_Info_t     **node_table;

    Evt_Node_t          *rhs;
    Evt_Node_t          *rhsold;

    Evt_Output_Queue_t  *output_queue;
    Evt_Output_Event_t  *output_event;

    Mif_Boolean_t       invert, equal;
    double              delay;

    output_queue = &(ckt->evt->queue.output);
    output_table = ckt->evt->info.output_table;
    node_table = ckt->evt->info.node_table;

    output_index = port->evt_data.output_index;
    node_index = output_table[output_index]->node_index;
    udn_index = node_table[node_index]->udn_index;
    invert = port->invert;

    /* if transient analysis, just put the output event on the queue */
    /* to be processed at a later time */

    if (g_mif_info.circuit.anal_type == MIF_TRAN) {
        delay = port->delay;
        if(delay <= 0.0) {
            printf("\nERROR - Output delay <= 0 not allowed - output ignored!\n");
            printf("  Instance: %s\n  Node: %s\n  Time: %f \n",
                   g_mif_info.instance->MIFname, node_table[node_index]->name,
                   g_mif_info.ckt->CKTtime);
            return;
        }
        /* Remove the (now used) struct from the port data struct. */

        output_event = port->next_event;
        port->next_event = NULL;

        /* Invert the output value if necessary */
        if(invert)
            g_evt_udn_info[udn_index]->invert
                (output_event->value);
        /* Add it to the queue */
        EVTqueue_output(ckt, output_index, udn_index, output_event,
                        g_mif_info.circuit.evt_step,
                        g_mif_info.circuit.evt_step + delay);
        return;
    } else {
        /* If not transient analysis, process immediately. */
        /* Determine if output has changed from rhsold value */
        /* and put entry in output queue changed list if so */

        rhs = ckt->evt->data.node->rhs;
        rhsold = ckt->evt->data.node->rhsold;

        /* Determine if changed */
        num_outputs = node_table[node_index]->num_outputs;
        if(num_outputs > 1) {
            output_subindex = output_table[output_index]->output_subindex;
            if(invert)
                g_evt_udn_info[udn_index]->invert
                    (rhs[node_index].output_value[output_subindex]);
            g_evt_udn_info[udn_index]->compare
                    (rhs[node_index].output_value[output_subindex],
                    rhsold[node_index].output_value[output_subindex],
                    &equal);
            if(! equal) {
                g_evt_udn_info[udn_index]->copy
                    (rhs[node_index].output_value[output_subindex],
                    rhsold[node_index].output_value[output_subindex]);
            }
        }
        else {
            if(invert)
                g_evt_udn_info[udn_index]->invert
                    (rhs[node_index].node_value);
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

        /* If changed, put in changed list of output queue */
        if(! equal) {
            if(! output_queue->changed[output_index]) {
                output_queue->changed[output_index] = MIF_TRUE;
                output_queue->changed_index[(output_queue->num_changed)++] =
                        output_index;
            }
        }

        return;

    } /* end else process immediately */
}
