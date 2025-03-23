/*============================================================================
FILE    EVTplot.c

MEMBER OF process XSPICE

Copyright 1992
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    5/7/92  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains function EVTplot which is used to provide basic
    plotting of event driven nodes through SPICE3's 'plot' command.

INTERFACES

    void EVTplot()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
//nclude "misc.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"
#include "ngspice/evtproto.h"
#include "ngspice/mif.h"
#include "ngspice/mifproto.h"

#include "ngspice/sim.h"
#include "ngspice/dvec.h"
#include "ngspice/cpstd.h"


/* Parse member qualifier from node name and find node index.
 * The node name may be qualified by a member name for nodes with
 * composite values such as Digital_t, as in "node_name(state)".
 */

int Evt_Parse_Node(const char *node, struct node_parse *result)
{
    Evt_Ckt_Data_t   *evt;
    CKTcircuit       *ckt;
    Evt_Node_Info_t **node_table;
    char             *name, *ptr;
    int               i, num_nodes;

    ckt = g_mif_info.ckt;
    if (!ckt)
        return -1;
    evt = ckt->evt;
    if (!evt)
        return -1;
    if (!evt->info.node_table)
        return -1;
    if (evt->counts.num_nodes == 0)
        return -1;

    /* Make a copy of the node name.  Do not free this string. */

    name = MIFcopy((char *)node);

    /* Convert to all lower case */

    strtolower(name);

    /* Divide into the node name and member name */

    result->node = name;
    for (ptr = name; *ptr != '\0'; ptr++)
        if (*ptr == '(')
            break;

    if (*ptr == '(') {
        *ptr = '\0';
        ptr++;
        result->member = ptr;
        for( ; *ptr != '\0'; ptr++)
            if (*ptr == ')')
                break;
        *ptr = '\0';
    } else {
        result->member = NULL;
    }

    /* Look for node name in the event-driven node list */

    node_table = evt->info.node_table;
    num_nodes = evt->counts.num_nodes;

    for (i = 0; i < num_nodes; i++) {
        if (cieq(name, node_table[i]->name))
            break;
    }
    if (i >= num_nodes) {
        tfree(name);
        return -1;
    }
    return i;
}

/*

EVTfindvec()

This function is called from FTE/vectors.c:findvec() when a node specified
for plotting cannot be located in the analog plot data.  It scans the
event driven data structures looking for the node, and if found, returns
a new 'dvec' structure holding the data to be plotted.  The dvec struct
is created with it's own v_scale member holding the event time vector
for this node since the time vector is not aligned with the analog data
points or with other event vectors.

The node name supplied as argument can either be a simple node name, or a
name of the form <node name>(<member>), where <member> is the member of
the event-driven structure to be plotted.  These member names are defined
by the individual "user-defined node" plot_val routines for the node
type in question.  If the simple node name form is used, the special
keyword "all" is supplied to the plot_val routine for the member name.

*/


struct dvec *EVTfindvec(
    char *node)  /* The node name (and optional member name) */
{
  char              *name;
  struct node_parse  result;
  int                index, i;
  int                udn_index;
  int                num_events;

  Evt_Ckt_Data_t    *evt;
  CKTcircuit        *ckt;
  Evt_Node_Info_t   *node_info;
  Evt_Node_t        *head;
  Evt_Node_t        *event;
 
  double *anal_point_vec;
  double *value_vec;
  double value = 0;

  struct dvec *d;
  struct dvec *scale;

  /* Exit immediately if event-driven stuff not allocated yet, */
  /* or if number of event nodes is zero. */

  index = Evt_Parse_Node(node, &result);
  if (index < 0)
      return NULL;
  name = result.node;
  ckt = g_mif_info.ckt;
  evt = ckt->evt;

  if (!evt->data.node) {
    tfree(name);
    return NULL;
  }

  /* Get the UDN type index */

  node_info = evt->info.node_table[index];
  udn_index = node_info->udn_index;

  /* Count the number of events */

  head = evt->data.node->head[index];
  for(event = head, num_events = 0; event; event = event->next)
    num_events++;

  /* Allocate arrays to hold the analysis point and node value vectors */
  anal_point_vec = TMALLOC(double, 2 * (num_events + 2));
  value_vec = TMALLOC(double, 2 * (num_events + 2));

  /* Iterate through the events and fill the arrays. */
  /* Note that we create vertical segments every time an event occurs. */

  for(i = 0, event = head; event; event = event->next) {

    /* If not first point, put the second value of the horizontal line in the vectors */
    if(i > 0) {
      anal_point_vec[i] = event->step;
      value_vec[i] = value;
      i++;
    }

    /* Get the next value by calling the appropriate UDN plot_val function */
    value = 0.0;
    g_evt_udn_info[udn_index]->plot_val(event->node_value,
                                        result.member ? result.member : "all",
                                        &value);

    /* Put the first value of the horizontal line in the vector */
    anal_point_vec[i] = event->step;
    value_vec[i] = value;
    i++;

  }

  /* Add one more point so that the line will extend to the end of the plot. */

  anal_point_vec[i] = ckt->CKTtime;
  value_vec[i++] = value;

  /* Allocate dvec structures and assign the vectors into them. */
  /* See FTE/OUTinterface.c:plotInit() for initialization example. */

  scale = dvec_alloc(tprintf("%s_steps", name),
                     SV_TIME,
                     (VF_REAL | VF_EVENT_NODE) & ~VF_PERMANENT,
                     i, anal_point_vec);

  d = dvec_alloc(name,
                 SV_VOLTAGE,
                 (VF_REAL | VF_EVENT_NODE) & ~VF_PERMANENT,
                 i, value_vec);

  d->v_scale = scale;


  /* Return the dvec */
  return(d);
}
