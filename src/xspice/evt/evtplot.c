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

/*saj for output */
#include "ngspice/sim.h"
#include "ngspice/dvec.h"
//#include "ftedata.h"
//#include "fteconstant.h"
//#include "util.h"
#include "ngspice/cpstd.h"


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
  char *name;
  char *member = "all";
  char *ptr;

  int  i;
  int  num_nodes;
  int  udn_index;
  int  num_events;

  Mif_Boolean_t     found;
  Evt_Ckt_Data_t   *evt;
  CKTcircuit       *ckt;
  Evt_Node_Info_t **node_table;
  Evt_Node_t       *head;
  Evt_Node_t       *event;
 
  double *anal_point_vec;
  double *value_vec;
  double value = 0;

  struct dvec *d;
  struct dvec *scale;

  /* Exit immediately if event-driven stuff not allocated yet, */
  /* or if number of event nodes is zero. */

  ckt = g_mif_info.ckt;
  if(! ckt)
    return(NULL);
  evt = ckt->evt;
  if(! evt)
    return(NULL);
  if(! evt->info.node_table)
    return(NULL);
  if(evt->counts.num_nodes == 0)
    return(NULL);

  /* Make a copy of the node name. */
  /* Do not free this string.  It is assigned into the dvec structure below. */
  name = MIFcopy(node);

  /* Convert to all lower case */
  strtolower(name);

  /* Divide into the node name and member name */
  for(ptr = name; *ptr != '\0'; ptr++)
    if(*ptr == '(')
      break;

  if(*ptr == '(') {
    *ptr = '\0';
    ptr++;
    member = ptr;
    for( ; *ptr != '\0'; ptr++)
      if(*ptr == ')')
        break;
    *ptr = '\0';
  }

  /* Look for node name in the event-driven node list */
  num_nodes = evt->counts.num_nodes;
  node_table = evt->info.node_table;

  for(i = 0, found = MIF_FALSE; i < num_nodes; i++) {
    if(cieq(name, node_table[i]->name)) {
      found = MIF_TRUE;
      break;
    }
  }

  if(! found) {
    tfree(name);
    return(NULL);
  }

  /* Get the UDN type index */
  udn_index = node_table[i]->udn_index;

  if (!evt->data.node) {
//    fprintf(stderr, "Warning: No event data available! \n   Simulation not yet run?\n");
    tfree(name);
    return(NULL);
  }

  /* Count the number of events */
  head = evt->data.node->head[i];

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
    g_evt_udn_info[udn_index]->plot_val (event->node_value,
                                              member,
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

  ptr = tprintf("%s_steps", name);
  scale = dvec_alloc(ptr,
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
