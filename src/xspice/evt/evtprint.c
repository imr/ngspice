/*============================================================================
FILE    EVTprint.c

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    7/11/2012  Holger Vogt   Replace printf by out_printf to allow output redirection

SUMMARY

    This file contains function EVTprint which is used to provide a simple
    tabular output of event-driven node data.  This printout is invoked
    through a new nutmeg command called 'eprint' which takes event-driven
    node names as argument.

INTERFACES

    void EVTprint(wordlist *wl)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"

#include "ngspice/cpstd.h"
#include "ngspice/cpextern.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"



static int get_index(char *node_name);

static void print_data(
    Mif_Boolean_t dcop,
    double        step,
    char          **node_value,
    int           nargs);

/*
EVTprint

This function implements the 'eprint' command used to print
event-driven node values and messages from the latest simulation.

This is a simple prototype implementation of the
eprint command for testing purposes.  It is currently lacking
in the following areas:

1)  It accepts only up to 16 nodes.

2)  It does not support the selected printing of different
    members of a user-defined data struct.

3)  It is dumb in its output formatting - just tries to print
    everything on a line with 4 space separators.

4)  It works only for the latest simulation - i.e. it does not
    use the evt jobs structure to find old results.

5)  It does not allow a range of timesteps to be selected.

*/



#define EPRINT_MAXARGS  32


void EVTprint(
    wordlist *wl)    /* The command line entered by user */
{

    int i;
    int nargs;
    int num_ports;

    wordlist    *w;

    char        *node_name[EPRINT_MAXARGS];
    int         node_index[EPRINT_MAXARGS];
    int         udn_index[EPRINT_MAXARGS];
    Evt_Node_t  *node_data[EPRINT_MAXARGS];
    char        *node_value[EPRINT_MAXARGS];

    CKTcircuit  *ckt;

    Evt_Node_Info_t  **node_table;
    Evt_Port_Info_t  **port_table;

    Mif_Boolean_t    more;
    Mif_Boolean_t    dcop;

    double      step = 0.0;
    double      next_step;
    double      this_step;

    char        *value;

    Evt_Msg_t   *msg_data;
    Evt_Statistic_t  *statistics;


    /* Count the number of arguments to the command */
    nargs = 0;
    w = wl;
    while(w) {
        nargs++;
        w = w->wl_next;
    }

    if(nargs < 1) {
        printf("Usage: eprint <node1> <node2> ...\n");
        return;
    }
    if(nargs > EPRINT_MAXARGS) {
        printf("ERROR - eprint currently limited to 32 arguments\n");
        return;
    }

    /* Get needed pointers */
    ckt = g_mif_info.ckt;
    node_table = ckt->evt->info.node_table;

    /* Get data for each argument */
    w = wl;
    for(i = 0; i < nargs; i++) {
        node_name[i] = w->wl_word;
        node_index[i] = get_index(node_name[i]);
        if(node_index[i] < 0) {
            printf("ERROR - Node %s is not an event node.\n", node_name[i]);
            return;
        }
        udn_index[i] = node_table[node_index[i]]->udn_index;
        if (ckt->evt->data.node)
            node_data[i] = ckt->evt->data.node->head[node_index[i]];
        else  {
            printf("ERROR - No node data: simulation not yet run?\n");
            return;
        }
        node_value[i] = "";
        w = w->wl_next;
    }

    out_init();

    /* Print results data */
    out_printf("\n**** Results Data ****\n\n");

    /* Print the column identifiers */
    out_printf("Time or Step\n");
    for(i = 0; i < nargs; i++)
        out_printf("%s\n",node_name[i]);
    out_printf("\n\n");

    /* Scan the node data and determine if the first vector */
    /* is for a DCOP analysis or the first step in a swept DC */
    /* or transient analysis.  Also, determine if there is */
    /* more data following it and if so, what the next step */
    /* is. */
    more = MIF_FALSE;
    dcop = MIF_FALSE;
    next_step = 1e30;
    for(i = 0; i < nargs; i++) {
        if(node_data[i]->op)
            dcop = MIF_TRUE;
        else
            step = node_data[i]->step;
        (*(g_evt_udn_info[udn_index[i]]->print_val))
                (node_data[i]->node_value, "all", &value);
        node_value[i] = value;
        node_data[i] = node_data[i]->next;
        if(node_data[i]) {
            more = MIF_TRUE;
            if(node_data[i]->step < next_step)
                next_step = node_data[i]->step;
        }
    }

    /* Print the data */
    print_data(dcop, step, node_value, nargs);

    /* While there is more data, get the next values and print */
    while(more) {

        more = MIF_FALSE;
        this_step = next_step;
        next_step = 1e30;

        for(i = 0; i < nargs; i++) {

            if(node_data[i]) {
                if(node_data[i]->step == this_step) {
                    (*(g_evt_udn_info[udn_index[i]]->print_val))
                            (node_data[i]->node_value, "all", &value);
                    node_value[i] = value;
                    node_data[i] = node_data[i]->next;
                }
                if(node_data[i]) {
                    more = MIF_TRUE;
                    if(node_data[i]->step < next_step)
                        next_step = node_data[i]->step;
                }

            } /* end if node_data not NULL */

        } /* end for number of args */

        print_data(MIF_FALSE, this_step, node_value, nargs);

    } /* end while there is more data */
    out_printf("\n\n");


    /* Print messages for all ports */
    out_printf("\n**** Messages ****\n\n");

    num_ports = ckt->evt->counts.num_ports;
    port_table = ckt->evt->info.port_table;

    for(i = 0; i < num_ports; i++) {

        /* Get pointer to messages for this port */
        msg_data = ckt->evt->data.msg->head[i];

        /* If no messages on this port, skip */
        if(! msg_data)
            continue;

        /* Print the port description */
        out_printf("Node: %s   Inst: %s   Conn: %s   Port: %d\n\n",
                port_table[i]->node_name,
                port_table[i]->inst_name,
                port_table[i]->conn_name,
                port_table[i]->port_num);

        /* Print the messages on this port */
        while(msg_data) {
            if(msg_data->op)
                printf("DCOP            ");
            else
                printf("%-16.9e", msg_data->step);
            printf("%s\n", msg_data->text);
            msg_data = msg_data->next;
        }
        out_printf("\n\n");

    } /* end for number of ports */


    /* Print statistics */
    out_printf("\n**** Statistics ****\n\n");

    statistics = ckt->evt->data.statistics;
    out_printf("Operating point analog/event alternations:  %d\n",
            statistics->op_alternations);
    out_printf("Operating point load calls:                 %d\n",
            statistics->op_load_calls);
    out_printf("Operating point event passes:               %d\n",
            statistics->op_event_passes);
    out_printf("Transient analysis load calls:              %d\n",
            statistics->tran_load_calls);
    out_printf("Transient analysis timestep backups:        %d\n",
            statistics->tran_time_backups);

    out_printf("\n\n");
}




/*
get_index

This function determines the index of a specified event-driven node.
*/


static int get_index(
    char *node_name      /* The name of the node to search for */
)
{

    /* Get the event-driven node index for the specified name */

    int                 index;

    Mif_Boolean_t       found;
    Evt_Node_Info_t     *node;
    CKTcircuit          *ckt;


    /* Scan list of nodes in event structure to see if there */

    found = MIF_FALSE;
    index = 0;

    ckt = g_mif_info.ckt;
    node = ckt->evt->info.node_list;

    while(node) {
        if(strcmp(node_name, node->name) == 0) {
            found = MIF_TRUE;
            break;
        }
        else {
            index++;
            node = node->next;
        }
    }

    /* Return the index or -1 if not found */
    if(! found)
        return(-1);
    else
        return(index);
}




/*
print_data

This function prints the values of one or more nodes to
standard output.
*/


static void print_data(
    Mif_Boolean_t dcop,          /* Is this the operating point data */
    double        step,          /* The analysis step if dcop */
    char          **node_value,  /* The array of values to be printed */
    int           nargs)         /* The size of the value array */
{

    int  i;
    char step_str[100];

    if(dcop)
        strcpy(step_str, "DCOP            ");
    else
        sprintf(step_str, "%-16.9e", step);

    out_printf("%s", step_str);
    for(i = 0; i < nargs; i++)
        out_printf("    %s", node_value[i]);
    out_printf("\n");
}
