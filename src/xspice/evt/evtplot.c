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

#include <stdio.h>
#include <string.h>

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

  Mif_Boolean_t   found;
  Evt_Node_Info_t **node_table;
  Evt_Node_t      *head;
  Evt_Node_t      *event;

  double *anal_point_vec;
  double *value_vec;
  double value = 0;

  struct dvec *d;
  struct dvec *scale;

  /* Exit immediately if event-driven stuff not allocated yet, */
  /* or if number of event nodes is zero. */
  if(! g_mif_info.ckt)
    return(NULL);
  if(! g_mif_info.ckt->evt)
    return(NULL);
  if(g_mif_info.ckt->evt->counts.num_nodes == 0)
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
  num_nodes = g_mif_info.ckt->evt->counts.num_nodes;
  node_table = g_mif_info.ckt->evt->info.node_table;

  for(i = 0, found = MIF_FALSE; i < num_nodes; i++) {
    if(cieq(name, node_table[i]->name)) {
      found = MIF_TRUE;
      break;
    }
  }

  if(! found)
    return(NULL);

  /* Get the UDN type index */
  udn_index = node_table[i]->udn_index;

  /* Count the number of events */
  head = g_mif_info.ckt->evt->data.node->head[i];

  for(event = head, num_events = 0; event; event = event->next)
    num_events++;

  /* Allocate arrays to hold the analysis point and node value vectors */
  anal_point_vec = TMALLOC(double, 2 * (num_events + 2));
  value_vec = TMALLOC(double, 2 * (num_events + 2));

  /* Iterate through the events and fill the arrays. */
  /* Note that we create vertical segments every time an event occurs. */
  /* Need to modify this in the future to complete the vector out to the */
  /* last analysis point... */
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

  /* Allocate dvec structures and assign the vectors into them. */
  /* See FTE/OUTinterface.c:plotInit() for initialization example. */

  scale = TMALLOC(struct dvec, 1);

  scale->v_name = MIFcopy("step");
  scale->v_type = SV_TIME;
  scale->v_flags = VF_REAL & ~VF_PERMANENT;
  scale->v_length = i;
  scale->v_realdata = anal_point_vec;
  scale->v_scale = NULL;

  d = TMALLOC(struct dvec, 1);

  d->v_name = name;
  d->v_type = SV_VOLTAGE;
  d->v_flags = VF_REAL & ~VF_PERMANENT;
  d->v_length = i;
  d->v_realdata = value_vec;
  d->v_scale = scale;


  /* Return the dvec */
  return(d);
}
