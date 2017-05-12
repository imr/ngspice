/*============================================================================
FILE    EVTshared.c

MEMBER OF process XSPICE

This code is in the public domain

AUTHORS

    5/12/17  Holger Vogt

MODIFICATIONS

    

SUMMARY

    This file function to prepare event node data for transfer over the
	shared ngspice interface.

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
#include "ngspice/fteext.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"

#include <time.h>


static int get_index(char *node_name);

struct evt_data {
	Mif_Boolean_t dcop;
	double        step;
	char          *node_value;
};

struct evt_shared_data {
	struct evt_data *evt_dect;
	int num_steps;
};

struct evt_data *return_node;
struct evt_shared_data *return_all;


struct evt_shared_data
*EVTshareddata(
    char *node_name)    /* The command line called by ngGet_EVT_Info(char* nodename) */
{

    int i;
    int nodes;

    int         node_index;
    int         udn_index;
    Evt_Node_t  *node_data, *count_data;
    char        *node_value;

    CKTcircuit  *ckt;

    Evt_Node_Info_t  **node_table;

    Mif_Boolean_t    more;
    Mif_Boolean_t    dcop;

    double      step = 0.0;
    double      next_step;
    double      this_step;

    char        *value;

    /* Get needed pointers */
    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return NULL;
    }
    node_table = ckt->evt->info.node_table;

    /* Get data for the node */
    node_index = get_index(node_name);
    if(node_index < 0) {
        fprintf(cp_err, "ERROR - Node %s is not an event node.\n", node_name);
        return NULL;
    }
    udn_index = node_table[node_index]->udn_index;
    if (ckt->evt->data.node)
        node_data = ckt->evt->data.node->head[node_index];
    else  {
        fprintf(cp_err, "ERROR - No node data: simulation not yet run?\n");
        return NULL;
    }
    node_value = "";

    /* Scan the node data and determine if the first vector */
    /* is for a DCOP analysis or the first step in a swept DC */
    /* or transient analysis.  Also, determine if there is */
    /* more data following it and if so, what the next step */
    /* is. */
    more = MIF_FALSE;
    dcop = MIF_FALSE;
    next_step = 1e30;

    if(node_data->op)
        dcop = MIF_TRUE;
    else
        step = node_data->step;
    (*(g_evt_udn_info[udn_index]->print_val))
            (node_data->node_value, "all", &value);
    node_value = value;
    node_data = node_data->next;
    if(node_data) {
        more = MIF_TRUE;
        if(node_data->step < next_step)
            next_step = node_data->step;
    }

    /* Count the neumber of node data */
	count_data = node_data;
	nodes = 0;
	while (count_data) {
		nodes++;
		count_data = count_data->next;
	}

	/* Store the data */
	return_node = TMALLOC(struct evt_data, nodes);
	return_node[0].dcop = dcop;
	return_node[0].node_value = copy(value);
	return_node[0].step = step;

    /* While there is more data, get the next values and print */
	i = 1;
	while(more) {

        more = MIF_FALSE;
        this_step = next_step;
        next_step = 1e30;

        if(node_data) {
            if(node_data->step == this_step) {
                (*(g_evt_udn_info[udn_index]->print_val))
                        (node_data->node_value, "all", &value);
                node_value = value;
                node_data = node_data->next;
            }
            if(node_data) {
                more = MIF_TRUE;
                if(node_data->step < next_step)
                    next_step = node_data->step;
            }

        } /* end if node_data not NULL */

		return_node[i].dcop = dcop;
		return_node[i].node_value = copy(value);
		return_node[i].step = this_step;
		i++;
    } /* end while there is more data */
	return_all = TMALLOC(struct evt_shared_data, 1);
	return_all->evt_dect = return_node;
	return_all->num_steps = i;
	return return_all;
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
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return(-1);
    }
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


