/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"

#include "ngspice/devdefs.h"
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"

#include "inpxx.h"
#include <stdio.h>

void INP2N(CKTcircuit *ckt, INPtables *tab, struct card *current) {
  /* Mname <node> <node> <node> <node> <model> [L=<val>]
   *       [W=<val>] [AD=<val>] [AS=<val>] [PD=<val>]
   *       [PS=<val>] [NRD=<val>] [NRS=<val>] [OFF]
   *       [IC=<val>,<val>,<val>]
   */

  int type;   /* the type the model says it is */
  char *line; /* the part of the current line left to parse */
  char *name; /* the resistor's name */
  // limit to at most 20 nodes
  const int max_i = 20;
  CKTnode *node[20];
  int error;         /* error code temporary */
  int numnodes;      /* flag indicating 4 or 5 (or 6 or 7) nodes */
  GENinstance *fast; /* pointer to the actual instance */
  int waslead;    /* flag to indicate that funny unlabeled number was found */
  double leadval; /* actual value of unlabeled number */
  INPmodel *thismodel; /* pointer to model description for user's model */
  GENmodel *mdfast;    /* pointer to the actual model */
  int i;

  line = current->line;

  INPgetNetTok(&line, &name, 1);
  INPinsert(&name, tab);

  for (i = 0;; i++) {
    char *token;
    INPgetNetTok(&line, &token, 1);

    if (i >= 2) {
      txfree(INPgetMod(ckt, token, &thismodel, tab));

      /* /1* check if using model binning -- pass in line since need 'l' and 'w' *1/ */
      /* if (!thismodel) */
      /*   txfree(INPgetModBin(ckt, token, &thismodel, tab, line)); */

      if (thismodel) {
        INPinsert(&token, tab);
        break;
      }
    }
    if (i >= max_i) {
      LITERR("could not find a valid modelname");
      return;
    }
    INPtermInsert(ckt, &token, tab, &node[i]);
  }

  type = thismodel->INPmodType;
  mdfast = thismodel->INPmodfast;
  IFdevice *dev = ft_sim->devices[type];

  if (!dev->registry_entry) {
    LITERR("incorrect model type! Expected OSDI device");
    return;
  }

  if (i == 0) {
    LITERR("not enough nodes");
    return;
  }

  if (i > *dev->terms) {
    LITERR("too many nodes connected to instance");
    return;
  }

  numnodes = i;

  IFC(newInstance, (ckt, mdfast, &fast, name));

  for (i = 0; i < *dev->terms; i++)
    if (i < numnodes)
      IFC(bindNode, (ckt, fast, i + 1, node[i]));
    else
      GENnode(fast)[i] = -1;

  PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
  if (waslead)
    LITERR(" error:  no unlabeled parameter permitted on osdi devices\n");
}
