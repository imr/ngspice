/*============================================================================
FILE    EVTdump.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    6/15/92  Bill Kuhn

MODIFICATIONS

    06/05/17  Holger Vogt Shared ngspice additions

SUMMARY

    This file contains functions used
    to send event-driven node results data to the IPC channel when
    the simulator is used with CAE software.
    It also sends data to a caller via callback, if shared ngspice
    is enabled.

INTERFACES

    EVTdump()
    EVTshareddump()

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
#include "ngspice/evtproto.h"
#include "ngspice/evtudn.h"

#include "ngspice/ipc.h"
#include "ngspice/ipctiein.h"
#include "ngspice/ipcproto.h"

#ifdef SHARED_MODULE
/* global flag, TRUE if callback is used */
extern bool wantevtdata;
extern void shared_send_event(int, double, double, char *, void *, int, int);
extern void shared_send_dict(int, int, char*, char*);
static void EVTshareddump(
    CKTcircuit    *ckt,  /* The circuit structure */
    Ipc_Anal_t    mode,  /* The analysis mode for this call */
    double        step);  /* The sweep step for a DCTRCURVE analysis, or */
                         /* 0.0 for DCOP and TRAN */

static void EVTsharedsend_line(
    int         ipc_index,         /* The index used in the dictionary */
    double      step,              /* The analysis step */
    void        *node_value,       /* The node value */
    int         udn_index,         /* The user-defined node index */
    int         mode);             /* mode (op, dc, tran) we are in */
#endif

static void EVTsend_line(
    int         ipc_index,         /* The index used in the dictionary */
    double      step,              /* The analysis step */
    void        *node_value,       /* The node value */
    int         udn_index);        /* The user-defined node index */



/*
EVTdump

This function is called to send event-driven node data to the IPC
channel.  A ``mode'' argument determines how the data is located in
the event data structure and what data is sent.

If the mode is DCOP, then this is necessarily the first call to
the function.  In this case, the set of event-driven nodes is
scanned to determine which should be sent.  Only nodes that are
not inside subcircuits are sent.  Next, the function sends
a ``dictionary'' of node names/types vs. node indexes.
Finally, the function sends the DC operating point solutions
for the event-driven nodes in the dictionary.

If the mode is DCTRCURVE, it is assumed that the function has
already been called with mode = DCOP.  The function scans the solution
vector and sends data for any nodes that have changed.

If the mode is TRAN, it is assumed that the function has already
been called once with mode = DCOP.  The function scans the
event data for nodes that have changed since the last accepted
analog timepoint and sends the new data.

Note:  This function must be called BEFORE calling EVTop_save or
EVTaccept() so that the state of the node data structure will allow
it to determine what has changed.
*/


typedef struct evtdump_s {
    Mif_Boolean_t  send;             /* True if this node should be sent */
    int            ipc_index;        /* Index for this node in dict sent to CAE system */
    char           *node_name_str;   /* Node name */
    char           *udn_type_str;    /* UDN type */
} evtdump_dict_t;



void EVTdump(
    CKTcircuit    *ckt,  /* The circuit structure */
    Ipc_Anal_t    mode,  /* The analysis mode for this call */
    double        step)  /* The sweep step for a DCTRCURVE analysis, or */
                         /* 0.0 for DCOP and TRAN */
{
    static evtdump_dict_t *node_dict = NULL;
    static int            num_send_nodes;

    int         i;
    int         j;
    int         num_nodes;
    int         num_modified;
    int         index;

    char        *name;
    int         name_len;

    Mif_Boolean_t       firstcall;

    Evt_Node_Data_t     *node_data;

    Evt_Node_t          *rhsold;
    Evt_Node_t          **head;
    Evt_Node_t          *here;

    Evt_Node_Info_t     **node_table;

    char                buff[10000];

    Mif_Boolean_t       equal;

#ifdef SHARED_MODULE
    if((! g_ipc.enabled) && (!wantevtdata))
        return;
    if ((!g_ipc.enabled) && (wantevtdata)) {
        EVTshareddump(ckt, mode, step);
        return;
    }
#else
    /* Return immediately if IPC is not enabled */
    if(! g_ipc.enabled)
        return;
#endif
    /* Get number of event-driven nodes */
    num_nodes = ckt->evt->counts.num_nodes;

    /* Exit immediately if no event-driven nodes in circuit */
    if(num_nodes <= 0)
        return;


    /* Get pointers for fast access to event data */
    node_data = ckt->evt->data.node;
    node_table = ckt->evt->info.node_table;
    rhsold = node_data->rhsold;
    head = node_data->head;


    /* Determine if this is the first call */
    if(node_dict == NULL)
        firstcall = MIF_TRUE;
    else
        firstcall = MIF_FALSE;


    /* If this is the first call, get the dictionary info */
    if(firstcall) {

        /* Allocate local data structure used to process nodes */
        node_dict = TMALLOC(evtdump_dict_t, num_nodes);

        /* Loop through all nodes to determine which nodes should be sent. */
        /* Only nodes not within subcircuits qualify. */

        num_send_nodes = 0;
        for(i = 0; i < num_nodes; i++) {

            /* Get the name of the node. */
            name = node_table[i]->name;

            /* If name is in a subcircuit, mark that node should not be sent */
            /* and continue to next node. */
            name_len = (int) strlen(name);
            for(j = 0; j < name_len; j++) {
                if(name[j] == ':')
                    break;
            }
            if(j < name_len) {
                node_dict[i].send = MIF_FALSE;
                continue;
            }

            /* Otherwise, fill in info in dictionary. */
            node_dict[i].send = MIF_TRUE;
            node_dict[i].ipc_index = num_send_nodes;
            node_dict[i].node_name_str = name;
            node_dict[i].udn_type_str = g_evt_udn_info[node_table[i]->udn_index]->name;

            /* Increment the count of nodes to be sent. */
            num_send_nodes++;
        } /* end for */
    } /* end if first call */


    /* Exit if there are no nodes to be sent */
    if(num_send_nodes <= 0)
        return;


    /* If this is the first call, send the dictionary */
    if(firstcall) {
        ipc_send_evtdict_prefix();
        for(i = 0; i < num_nodes; i++) {
            if(node_dict[i].send) {
                sprintf(buff, "%d %s %s", node_dict[i].ipc_index,
                                          node_dict[i].node_name_str,
                                          node_dict[i].udn_type_str);
                ipc_send_line(buff);
            }
        }
        ipc_send_evtdict_suffix();
    }

    /* If this is the first call, send the operating point solution */
    /* and return. */
    if(firstcall) {
        ipc_send_evtdata_prefix();
        for(i = 0; i < num_nodes; i++) {
            if(node_dict[i].send) {
                EVTsend_line(node_dict[i].ipc_index,
                             step,
                             rhsold[i].node_value,
                             node_table[i]->udn_index);
            }
        }
        ipc_send_evtdata_suffix();
        return;
    }

    /* Otherwise, this must be DCTRCURVE or TRAN mode and we need to */
    /* send only stuff that has changed since the last call. */
    /* The determination of what to send is modeled after code in */
    /* EVTop_save() for DCTRCURVE and EVTaccept() for TRAN. */

    if(mode == IPC_ANAL_DCTRCURVE) {
        /* Send data prefix */
        ipc_send_evtdata_prefix();
        /* Loop through event nodes */
        for(i = 0; i < num_nodes; i++) {
            /* If dictionary indicates this node should be sent */
            if(node_dict[i].send) {
                /* Locate end of node data */
                here = head[i];
                for(;;) {
                    if(here->next)
                        here = here->next;
                    else
                        break;
                }
                /* Compare entry at end of list to rhsold */
                g_evt_udn_info[node_table[i]->udn_index]->compare (
                          rhsold[i].node_value,
                          here->node_value,
                          &equal);
                /* If value in rhsold is different, send it */
                if(!equal) {
                    EVTsend_line(node_dict[i].ipc_index,
                                 step,
                                 rhsold[i].node_value,
                                 node_table[i]->udn_index);
                }
            }
        }
        /* Send data suffix and return */
        ipc_send_evtdata_suffix();
        return;
    }


    if(mode == IPC_ANAL_TRAN) {
        /* Send data prefix */
        ipc_send_evtdata_prefix();
        /* Loop through list of nodes modified since last time */
        num_modified = node_data->num_modified;
        for(i = 0; i < num_modified; i++) {
            /* Get the index of the node modified */
            index = node_data->modified_index[i];
            /* If dictionary indicates this node should be sent */
            if(node_dict[index].send) {
                /* Scan through new events and send the data for each event */
                here = *(node_data->last_step[index]);
                while((here = here->next) != NULL) {
                    EVTsend_line(node_dict[index].ipc_index,
                                 here->step,
                                 here->node_value,
                                 node_table[index]->udn_index);
                }
            }
        }
        /* Send data suffix and return */
        ipc_send_evtdata_suffix();
        return;
    }

}



/*
EVTsend_line

This function formats the event node data and sends it to the IPC channel.
*/


static void EVTsend_line(
    int         ipc_index,         /* The index used in the dictionary */
    double      step,              /* The analysis step */
    void        *node_value,       /* The node value */
    int         udn_index)         /* The user-defined node index */
{
    double dvalue;
    char   *svalue;
    void   *pvalue;
    int    len;

    /* Get the data to send */
    if(g_evt_udn_info[udn_index]->plot_val)
        g_evt_udn_info[udn_index]->plot_val (node_value, "", &dvalue);
    else
        dvalue = 0.0;

    if(g_evt_udn_info[udn_index]->print_val)
        g_evt_udn_info[udn_index]->print_val (node_value, "", &svalue);
    else
        svalue = "";

    if(g_evt_udn_info[udn_index]->ipc_val)
        g_evt_udn_info[udn_index]->ipc_val (node_value, &pvalue, &len);
    else {
        pvalue = NULL;
        len = 0;
    }

    /* Send it to the IPC channel */
    ipc_send_event(ipc_index, step, dvalue, svalue, pvalue, len);
}


#ifdef SHARED_MODULE
static void EVTshareddump(
    CKTcircuit    *ckt,  /* The circuit structure */
    Ipc_Anal_t    mode,  /* The analysis mode for this call */
    double        step)  /* The sweep step for a DCTRCURVE analysis, or */
                         /* 0.0 for DCOP and TRAN */
{
    static evtdump_dict_t *node_dict = NULL;
    static int            num_send_nodes;

    int         i;
    int         j;
    int         num_nodes;
    int         num_modified;
    int         index;

    char        *name;
    int         name_len;

    Mif_Boolean_t       firstcall;

    Evt_Node_Data_t     *node_data;

    Evt_Node_t          *rhsold;
    Evt_Node_t          **head;
    Evt_Node_t          *here;

    Evt_Node_Info_t     **node_table;

    Mif_Boolean_t       equal;

    /* Get number of event-driven nodes */
    num_nodes = ckt->evt->counts.num_nodes;

    /* Exit immediately if no event-driven nodes in circuit */
    if (num_nodes <= 0)
        return;


    /* Get pointers for fast access to event data */
    node_data = ckt->evt->data.node;
    node_table = ckt->evt->info.node_table;
    rhsold = node_data->rhsold;
    head = node_data->head;


    /* Determine if this is the first call */
    if (node_dict == NULL)
        firstcall = MIF_TRUE;
    else
        firstcall = MIF_FALSE;


    /* If this is the first call, get the dictionary info */
    if (firstcall) {

        /* Allocate local data structure used to process nodes */
        node_dict = TMALLOC(evtdump_dict_t, num_nodes);

        /* Loop through all nodes to determine which nodes should be sent. */
        /* Only nodes not within subcircuits qualify. */

        num_send_nodes = 0;
        for (i = 0; i < num_nodes; i++) {

            /* Get the name of the node. */
            name = node_table[i]->name;

            /* If name is in a subcircuit, mark that node should not be sent */
            /* and continue to next node. */
            name_len = (int)strlen(name);
            for (j = 0; j < name_len; j++) {
                if (name[j] == ':')
                    break;
            }
            if (j < name_len) {
                node_dict[i].send = MIF_FALSE;
                continue;
            }

            /* Otherwise, fill in info in dictionary. */
            node_dict[i].send = MIF_TRUE;
            node_dict[i].ipc_index = num_send_nodes;
            node_dict[i].node_name_str = name;
            node_dict[i].udn_type_str = g_evt_udn_info[node_table[i]->udn_index]->name;

            /* Increment the count of nodes to be sent. */
            num_send_nodes++;
        } /* end for */
    } /* end if first call */

        /* Exit if there are no nodes to be sent */
    if (num_send_nodes <= 0)
        return;

    /* If this is the first call, send the dictionary (the list of event nodes, line by line) */
    if (firstcall) {
        for (i = 0; i < num_nodes; i++) {
            if (node_dict[i].send)
                shared_send_dict(node_dict[i].ipc_index, num_nodes, node_dict[i].node_name_str, node_dict[i].udn_type_str);
        }
    }

    /* If this is the first call, send the operating point solution */
    /* and return. */
    if (firstcall) {
        for (i = 0; i < num_nodes; i++) {
            if (node_dict[i].send) {
                EVTsharedsend_line(node_dict[i].ipc_index,
                    step,
                    rhsold[i].node_value,
                    node_table[i]->udn_index,
                    mode);
            }
        }
        return;
    }

    /* Otherwise, this must be DCTRCURVE or TRAN mode and we need to */
    /* send only stuff that has changed since the last call. */
    /* The determination of what to send is modeled after code in */
    /* EVTop_save() for DCTRCURVE and EVTaccept() for TRAN. */

    if (mode == IPC_ANAL_DCTRCURVE) {
        /* Send data prefix */
        /* Loop through event nodes */
        for (i = 0; i < num_nodes; i++) {
            /* If dictionary indicates this node should be sent */
            if (node_dict[i].send) {
                /* Locate end of node data */
                here = head[i];
                for (;;) {
                    if (here->next)
                        here = here->next;
                    else
                        break;
                }
                /* Compare entry at end of list to rhsold */
                g_evt_udn_info[node_table[i]->udn_index]->compare(
                    rhsold[i].node_value,
                    here->node_value,
                    &equal);
                /* If value in rhsold is different, send it */
                if (!equal) {
                    EVTsharedsend_line(node_dict[i].ipc_index,
                        step,
                        rhsold[i].node_value,
                        node_table[i]->udn_index,
                        mode);
                }
            }
        }
        return;
    }


    if (mode == IPC_ANAL_TRAN) {
        /* Loop through list of nodes modified since last time */
        num_modified = node_data->num_modified;
        for (i = 0; i < num_modified; i++) {
            /* Get the index of the node modified */
            index = node_data->modified_index[i];
            /* If dictionary indicates this node should be sent */
            if (node_dict[index].send) {
                /* Scan through new events and send the data for each event */
                here = *(node_data->last_step[index]);
                while ((here = here->next) != NULL) {
                    EVTsharedsend_line(node_dict[index].ipc_index,
                        here->step,
                        here->node_value,
                        node_table[index]->udn_index,
                        mode);
                }
            }
        }
        return;
    }

}

/*
EVTsharedsend_line

This function formats the event node data and sends it to the caller via sharedspice.c.
*/


static void EVTsharedsend_line(
    int         dict_index,        /* The index used in the dictionary */
    double      step,              /* The analysis step */
    void        *node_value,       /* The node value */
    int         udn_index,         /* The user-defined node index */
    int         mode)              /* the mode (op, dc, tran) we are in */
{
    double dvalue;
    char   *svalue;
    void   *pvalue;
    int    len;

    /* Get the data to send */
    if (g_evt_udn_info[udn_index]->plot_val)
        g_evt_udn_info[udn_index]->plot_val(node_value, "", &dvalue);
    else
        dvalue = 0.0;

    if (g_evt_udn_info[udn_index]->print_val)
        g_evt_udn_info[udn_index]->print_val(node_value, "", &svalue);
    else
        svalue = "";

    if (g_evt_udn_info[udn_index]->ipc_val)
        g_evt_udn_info[udn_index]->ipc_val(node_value, &pvalue, &len);
    else {
        pvalue = NULL;
        len = 0;
    }

    /* Send it to sharedspice.c */
    shared_send_event(dict_index, step, dvalue, svalue, pvalue, len, mode);
}


#endif
