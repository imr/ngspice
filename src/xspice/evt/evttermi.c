/*============================================================================
FILE    EVTtermInsert.c

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

    This file contains function EVTtermInsert which is called by
    MIF_INP2A during the parsing of the input deck. EVTtermInsert is
    similar to SPICE3's INPtermInsert except that it is used when the node
    type is event-driven.  Calls to this function build the info lists
    for instances, nodes, outputs, and ports.  The completion of the info
    struct is carried out by EVTinit following the parsing of all
    instances in the deck.

INTERFACES

    void EVTtermInsert(
        CKTcircuit      *ckt,
        MIFinstance     *fast,
        char            *node_name,
        char            *type_name,
        int             conn_num,
        int             port_num,
        char            **err_msg)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
//#include "misc.h"

#include "ngspice/cktdefs.h"
//#include "util.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/mifproto.h"
#include "ngspice/evtproto.h"


static void EVTinst_insert(
    CKTcircuit    *ckt,
    MIFinstance   *fast,
    int           *inst_index,
    char          **err_msg);

static void EVTnode_insert(
    CKTcircuit  *ckt,
    MIFinstance *fast,
    int         inst_index,
    char        *node_name,
    char        *type_name,
    int         conn_num,
    int         port_num,
    int         *node_index,
    int         *output_subindex,
    char        **err_msg);

static void EVTport_insert(
    CKTcircuit  *ckt,
    MIFinstance *fast,
    int         inst_index,
    int         node_index,
    char        *node_name,
    int         conn_num,
    int         port_num,
    int         *port_index,
    char        **err_msg);

static void EVToutput_insert(
    CKTcircuit  *ckt,
    MIFinstance *fast,
    int         inst_index,
    int         node_index,
    int         port_index,
    int         output_subindex,
    int         conn_num,
    int         port_num,
    char        **err_msg);


/*
EVTtermInsert

This function is called by MIF_INP2A during the parsing of the input
deck. EVTtermInsert is similar to 3C1's INPtermInsert except that it is
used when the node type is event-driven.  Calls to this function build
the info lists for instances, nodes, outputs, and ports.  The
completion of the info struct is carried out by EVTinit following
the parsing of all instances in the deck.
*/


void EVTtermInsert(
    CKTcircuit      *ckt,         /* The circuit structure */
    MIFinstance     *fast,        /* The instance being parsed */
    char            *node_name,   /* The node name */
    char            *type_name,   /* The type of node */
    int             conn_num,     /* The port connection number */
    int             port_num,     /* The sub-port number - 0 for scalar ports */
    char            **err_msg)    /* Returned error message if any */
{

    int         inst_index;
    int         node_index;
    int         port_index;

    int         output_subindex;


    /* Get the instance index and create new entry in inst */
    /* info list if this is a new instance. */
    EVTinst_insert(ckt, fast, &inst_index, err_msg);
    if(*err_msg)
        return;

    /* Get the node index and create new entry in node info */
    /* list if this is a new node */
    EVTnode_insert(ckt, fast, inst_index, node_name, type_name,
                   conn_num, port_num, &node_index, &output_subindex,
                   err_msg);
    if(*err_msg)
        return;

    /* Create new entry in port info list and return port index */
    EVTport_insert(ckt, fast, inst_index, node_index, node_name, conn_num,
                   port_num, &port_index, err_msg);
    if(*err_msg)
        return;

    /* Create new entry in output info list if appropriate */
    if(fast->conn[conn_num]->is_output) {
        EVToutput_insert(ckt, fast, inst_index, node_index, port_index,
                   output_subindex, conn_num, port_num, err_msg);
        if(*err_msg)
            return;
    }
}




/*
EVTinst_insert

This function locates or creates a new entry for the specified
instance in the event-driven ``info'' structures during parsing.
*/


static void EVTinst_insert(
    CKTcircuit    *ckt,         /* The circuit structure */
    MIFinstance   *fast,        /* The instance being parsed */
    int           *inst_index,  /* The index found or added */
    char          **err_msg)    /* Error message if any */
{

    Mif_Boolean_t   found;
    int             index;

    Evt_Inst_Info_t *inst;
    Evt_Inst_Info_t **inst_ptr;

    NG_IGNORE(err_msg);


    /* Scan list of instances in event structure to see if already there */
    /* and get the index */
    found = MIF_FALSE;
    index = 0;
    inst = ckt->evt->info.inst_list;
    inst_ptr  = &(ckt->evt->info.inst_list);

    while(inst) {
        if(inst->inst_ptr == fast) {
            found = MIF_TRUE;
            break;
        }
        else {
            index++;
            inst_ptr = &(inst->next);
            inst = inst->next;
        }
    }


    /* If not found, create a new entry in list and increment the */
    /* instance count in the event structure */
    if(! found) {
        *inst_ptr = TMALLOC(Evt_Inst_Info_t, 1);
        inst = *inst_ptr;
        inst->next = NULL;
        inst->inst_ptr = fast;
        index = ckt->evt->counts.num_insts;
        (ckt->evt->counts.num_insts)++;
    }

    /* Record the inst index in the MIFinstance structure and return it */
    fast->inst_index = index;
    *inst_index = index;
}



/*
EVTnode_insert

This function locates or creates a new entry for the specified
node in the event-driven ``info'' structures during parsing.
*/


static void EVTnode_insert(
    CKTcircuit  *ckt,               /* The circuit structure */
    MIFinstance *fast,              /* The instance being parsed */
    int         inst_index,         /* The index of inst in evt structures */
    char        *node_name,         /* The node name */
    char        *type_name,         /* The node type specified */
    int         conn_num,           /* The port connection number */
    int         port_num,           /* The sub-port number - 0 if scalar port */
    int         *node_index,        /* The node index found or added */
    int         *output_subindex,   /* The output number on this node */
    char        **err_msg)          /* Error message text if any */
{

    int             i;
    int             udn_index=0;
    Mif_Boolean_t   found;

    int             index;

    Evt_Node_Info_t *node;
    Evt_Node_Info_t **node_ptr;

    Evt_Inst_Index_t *inst;
    Evt_Inst_Index_t **inst_ptr;


    /* *************************************** */
    /* Get and check the node type information */
    /* *************************************** */

    /* Scan the list of user-defined node types and get the index */
    found = MIF_FALSE;
    for(i = 0; i < g_evt_num_udn_types; i++) {
        if(strcmp(type_name, g_evt_udn_info[i]->name) == 0) {
            udn_index = i;
            found = MIF_TRUE;
            break;
        }
    }

    /* Report error if not recognized */
    if(! found) {
        *err_msg = "Unrecognized connection type";
        return;
    }

    /* If inverted, check to be sure invert function exists for type */
    if(fast->conn[conn_num]->port[port_num]->invert) {
        if(g_evt_udn_info[udn_index]->invert == NULL) {
            *err_msg = "Connection type cannot be inverted";
            return;
        }
    }


    /* ******************************************* */
    /* Find/create entry in event-driven node list */
    /* ******************************************* */

    /* Scan list of nodes in event structure to see if already there */
    /* and get the index */
    found = MIF_FALSE;
    index = 0;
    node = ckt->evt->info.node_list;
    node_ptr  = &(ckt->evt->info.node_list);

    while(node) {
        if(strcmp(node_name, node->name) == 0) {
            found = MIF_TRUE;
            break;
        }
        else {
            index++;
            node_ptr = &(node->next);
            node = node->next;
        }
    }


    /* If found, verify that connection type is same as type of node */
    if(found) {
        if(udn_index != node->udn_index) {
            *err_msg = "Node cannot have two different types";
            return;
        }
    }

    /* If not found, create a new entry in list and increment the */
    /* node count in the event structure */
    if(! found) {
        *node_ptr = TMALLOC(Evt_Node_Info_t, 1);
        node = *node_ptr;
        node->next = NULL;
        node->name = MIFcopy(node_name);
        node->udn_index = udn_index;
        node->save = MIF_TRUE; /* Backward compatible behaviour: save all. */
        index = ckt->evt->counts.num_nodes;
        (ckt->evt->counts.num_nodes)++;
    }


    /* ******************************************* */
    /* Update remaining items in node list struct  */
    /* ******************************************* */

    /* Update flag on node that indicates if inversion is used by any */
    /* instance inputs */
    if(fast->conn[conn_num]->is_input)
        if(! node->invert)
            node->invert = fast->conn[conn_num]->port[port_num]->invert;

    /* Increment counts of ports, outputs connected to node */
    (node->num_ports)++;
    if(fast->conn[conn_num]->is_output)
        (node->num_outputs)++;

    /* If this is an input, add instance to list if not already there */
    if(fast->conn[conn_num]->is_input) {

        found = MIF_FALSE;
        inst = node->inst_list;
        inst_ptr = &(node->inst_list);

        while(inst) {
            if(inst_index == inst->index) {
                found = MIF_TRUE;
                break;
            }
            else {
                inst_ptr = &(inst->next);
                inst = inst->next;
            }
        }

        if(! found) {
            (node->num_insts)++;
            *inst_ptr = TMALLOC(Evt_Inst_Index_t, 1);
            inst = *inst_ptr;
            inst->next = NULL;
            inst->index = inst_index;
        }
    }

    /* Record the node index in the MIFinstance structure */
    fast->conn[conn_num]->port[port_num]->evt_data.node_index = index;

    /* Return the node index */
    *node_index = index;
    if(fast->conn[conn_num]->is_output)
        *output_subindex = node->num_outputs - 1;
    else
        *output_subindex = 0;  /* just for safety - shouldn't need this */
}



/*
EVTport_insert

This function locates or creates a new entry for the specified
port in the event-driven ``info'' structures during parsing.
*/


static void EVTport_insert(
    CKTcircuit  *ckt,          /* The circuit structure */
    MIFinstance *fast,         /* The instance being parsed */
    int         inst_index,    /* The index of inst in evt structures */
    int         node_index,    /* The index of the node in evt structures */
    char        *node_name,    /* The node name */
    int         conn_num,      /* The port connection number */
    int         port_num,      /* The sub-port number - 0 if scalar port */
    int         *port_index,   /* The port index found or added */
    char        **err_msg)     /* Error message text if any */
{

    Evt_Port_Info_t     *port;
    Evt_Port_Info_t     **port_ptr;

    int                 index;

    NG_IGNORE(err_msg);

    /* Find the end of the port info list */
    port = ckt->evt->info.port_list;
    port_ptr = &(ckt->evt->info.port_list);

    index = 0;
    while(port) {
        port_ptr = &(port->next);
        port = port->next;
        index++;
    }


    /* Update the port count and create a new entry in the list */

    (ckt->evt->counts.num_ports)++;

    *port_ptr = TMALLOC(Evt_Port_Info_t, 1);
    port = *port_ptr;

    /* Fill in the elements */
    port->next = NULL;
    port->inst_index = inst_index;
    port->node_index = node_index;
    port->node_name  = MIFcopy(node_name);
    port->inst_name  = MIFcopy(fast->MIFname);
    port->conn_name  = MIFcopy(fast->conn[conn_num]->name);
    port->port_num   = port_num;

    /* Record the port index in the MIFinstance structure */
    fast->conn[conn_num]->port[port_num]->evt_data.port_index = index;

    /* Return the port index */
    *port_index = index;
}




/*
EVToutput_insert

This function locates or creates a new entry for the specified
output in the event-driven ``info'' structures during parsing.
*/


static void EVToutput_insert(
    CKTcircuit  *ckt,             /* The circuit structure */
    MIFinstance *fast,            /* The instance being parsed */
    int         inst_index,       /* The index of inst in evt structures */
    int         node_index,       /* The index of the node in evt structures */
    int         port_index,       /* The index of the port in the evt structures */
    int         output_subindex,  /* The output on this node */
    int         conn_num,         /* The port connection number */
    int         port_num,         /* The sub-port number - 0 if scalar port */
    char        **err_msg)        /* Error message text if any */
{
    Evt_Output_Info_t     *output;
    Evt_Output_Info_t     **output_ptr;

    int                   index;

    NG_IGNORE(err_msg);

    /* Find the end of the port info list */
    output = ckt->evt->info.output_list;
    output_ptr = &(ckt->evt->info.output_list);

    index = 0;
    while(output) {
        output_ptr = &(output->next);
        output = output->next;
        index++;
    }


    /* Update the port count and create a new entry in the list */

    (ckt->evt->counts.num_outputs)++;

    *output_ptr = TMALLOC(Evt_Output_Info_t, 1);
    output = *output_ptr;

    /* Fill in the elements */
    output->next = NULL;
    output->inst_index = inst_index;
    output->node_index = node_index;
    output->port_index = port_index;
    output->output_subindex = output_subindex;

    /* Record the output index and subindex in the MIFinstance structure */
    fast->conn[conn_num]->port[port_num]->evt_data.output_index = index;
    fast->conn[conn_num]->port[port_num]->evt_data.output_subindex
             = output_subindex;

}
