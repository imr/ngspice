/*============================================================================
FILE    EVTshared.c

MEMBER OF process XSPICE

This code is in the public domain.

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
#include "ngspice/sharedspice.h"
#include "ngspice/cpstd.h"
#include "ngspice/cpextern.h"
#include "ngspice/fteext.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"
#include "ngspice/evtshared.h"

#include <time.h>


static int get_index(char *node_name);
/*

// typedefs are done in sharedspice.h

typedef struct evt_data {
    Mif_Boolean_t dcop;
    double        step;
    char          *node_value;
} evt_data, *pevt_data;

typedef struct evt_shared_data {
    pevt_data evt_dect;
    int num_steps;
} evt_shared_data, *pevt_shared_data;
*/

pevt_data *return_node;
pevt_shared_data return_all;

/* delete the information return structures */
static void
delete_ret(void)
{
    int i;
    if (return_all) {
        for (i = 0; i < return_all->num_steps; i++) {
            tfree(return_all->evt_dect[i]->node_value);
            tfree(return_all->evt_dect[i]);
        }
        tfree(return_all);
    }
}

pevt_shared_data
EVTshareddata(
    char *node_name)    /* The command called by ngGet_EVT_Info(char* nodename) */
{

    int i;
    int num_points;

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

    delete_ret();

    /* just return if only deletion of previous data is requested */
    if (!node_name)
        return NULL;

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

    /* Count the number of data points of this node */
    count_data = node_data;
    num_points = 0;
    while (count_data) {
        num_points++;
        count_data = count_data->next;
    }

    /* Store the data */
    return_node = TMALLOC(pevt_data, num_points + 1);
    pevt_data newnode = TMALLOC(evt_data, 1);
    newnode->dcop = dcop;
    newnode->node_value = copy(value);
    newnode->step = step;
    return_node[0] = newnode;

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

        newnode = TMALLOC(evt_data, 1);
        newnode->dcop = dcop;
        newnode->node_value = copy(value);
        newnode->step = this_step;
        return_node[i] = newnode;
        i++;
    } /* end while there is more data */
    return_all = TMALLOC(evt_shared_data, 1);
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


char** EVTallnodes(void)
{
    static char** allnodes;
    int len = 0, i = 0;
    Evt_Node_Info_t  *node;
    CKTcircuit       *ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return NULL;
    }
    if (allnodes)
        tfree(allnodes);
    node = ckt->evt->info.node_list;
    /* count the event nodes */
    while (node) {
        len++;
        node = node->next;
    }
    if (len == 0) {
        fprintf(cp_err, "Error: no event nodes found.\n");
        return NULL;
    }
    allnodes = TMALLOC(char*, len + 1);
    node = ckt->evt->info.node_list;
    for (i = 0; i < len; i++) {
        allnodes[i] = node->name;
        node = node->next;
    }
    allnodes[len] = NULL;
    return allnodes;
}
