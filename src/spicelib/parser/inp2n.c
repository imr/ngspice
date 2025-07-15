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

  int          type;      /* Model type. */
  char        *line;      /* Unparsed part of the current line. */
  char        *name;      /* Device instance name. */
  int          error;     /* Temporary error code. */
  int          numnodes;  /* Flag indicating 4 or 5 (or 6 or 7) nodes. */
  GENinstance *fast;      /* Pointer to the actual instance. */
  int          waslead;   /* Funny unlabeled number was found. */
  double       leadval;   /* Value of unlabeled number. */
  INPmodel    *thismodel; /* Pointer to model description for user's model. */
  GENmodel    *mdfast;    /* Pointer to the actual model. */
  IFdevice    *dev;
  CKTnode     *node;
  char        *c, *token = NULL, *prev = NULL, *pprev = NULL, *eqp;
  int          i;

  line = current->line;
  INPgetNetTok(&line, &name, 1);
  INPinsert(&name, tab);

  /* Find the last non-parameter token in the line. */

  c = line;
  for (i = 0, eqp = NULL; *c != '\0'; ++i) {
      tfree(pprev);
      pprev = prev;
      prev = token;
      token = gettok_instance(&c);
      eqp = strchr(token, '=');
      if (eqp)
          break;
  }
  if (eqp) {
      tfree(token); // A parameter or starts with '='.
      if (*c == '=') {
          /* Now prev points to a parameter pprev is the model. */

          --i;
          token = pprev;
          tfree(prev);
      } else {
          token = prev;
          tfree(pprev);
      }
  }

  /* We have single terminal Verilog-A modules */

  if (i >= 2) {
      c = INPgetMod(ckt, token, &thismodel, tab);
      if (c) {
          LITERR(c);
          tfree(c);
          tfree(token);
          return;
      }
  }
  tfree(token);
  if (i < 2 || !thismodel) {
      LITERR("could not find a valid modelname");
      return;
  }
  type = thismodel->INPmodType;
  mdfast = thismodel->INPmodfast;
  dev = ft_sim->devices[type];

  if (!dev->registry_entry) {
    LITERR("incorrect model type! Expected OSDI device");
    return;
  }

  numnodes = i - 1;
  if (numnodes > *dev->terms) {
    LITERR("too many nodes connected to instance");
    return;
  }

  IFC(newInstance, (ckt, mdfast, &fast, name));

  /* Rescan to process nodes. */

  for (i = 0; i < *dev->terms; i++) {
      if (i < numnodes) {
          token = gettok_instance(&line);
          INPtermInsert(ckt, &token, tab, &node); // Consumes token
          IFC(bindNode, (ckt, fast, i + 1, node));
      } else {
          GENnode(fast)[i] = -1;
      }
  }
  token = gettok_instance(&line); // Eat model name.
  tfree(token);
  PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
  if (waslead)
    LITERR(" error:  no unlabeled parameter permitted on osdi devices\n");
}
