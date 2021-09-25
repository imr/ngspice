/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "ngspice/compatmode.h"
#include "inpxx.h"

void INP2C(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* parse a capacitor card */
/* Cname <node> <node> [<val>] [<mname>] [IC=<val>] */

    static int mytype = -1; /* the type we determine capacitors are */
    int type = 0;        /* the type the model says it is */
    char *line;          /* the part of the current line left to parse */
    char *saveline;      /* ... just in case we need to go back... */
    char *name;          /* the resistor's name */
    char *model;         /* the name of the capacitor's model */
    char *nname1;        /* the first node's name */
    char *nname2;        /* the second node's name */
    CKTnode *node1;      /* the first node's node pointer */
    CKTnode *node2;      /* the second node's node pointer */
    double val;          /* temp to held resistance */
    int error;           /* error code temporary */
    int error1;          /* secondary error code temporary */
    INPmodel *thismodel; /* pointer to model structure describing our model */
    GENmodel *mdfast = NULL; /* pointer to the actual model */
    GENinstance *fast;   /* pointer to the actual instance */
    IFvalue ptemp;       /* a value structure to package resistance into */
    int waslead;         /* flag to indicate that funny unlabeled number was found */
    double leadval;      /* actual value of unlabeled number */
    IFuid uid;           /* uid for default cap model */

#ifdef TRACE
    printf("In INP2C, Current line: %s\n", current->line);
#endif

    if (mytype < 0) {
        if ((mytype = INPtypelook("Capacitor")) < 0) {
            LITERR("Device type Capacitor not supported by this binary\n");
            return;
        }
    }
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);

    /* enable reading values like 4u7 */
    if (newcompat.lt)
        val = INPevaluateRKM_C(&line, &error1, 1);	/* [<val>] */
    else
        val = INPevaluate(&line, &error1, 1);	/* [<val>] */
    
    saveline = line;
    
    INPgetNetTok(&line, &model, 1);
    
    if (*model && (strcmp(model, "c") != 0)) {
    /* token isn't null */
      if (INPlookMod(model)) {
          /* If this is a valid model connect it */
          INPinsert(&model, tab);
          current->error = INPgetMod(ckt, model, &thismodel, tab);
          if (thismodel != NULL) {
          if (mytype != thismodel->INPmodType) {
              LITERR("incorrect model type");
              return;
          }
          mdfast = thismodel->INPmodfast;
          type = thismodel->INPmodType;
          }
      } else {
          tfree(model);
          /* It is not a model */
          line = saveline;    /* go back */
          type = mytype;
          if (!tab->defCmod) {    /* create default C model */
          IFnewUid(ckt, &uid, NULL, "C", UID_MODEL, NULL);
          IFC(newModel, (ckt, type, &(tab->defCmod), uid));
          }
          mdfast = tab->defCmod;
      }
      IFC(newInstance, (ckt, mdfast, &fast, name));
    } else {
      tfree(model);
      /* The token is null and a default model will be created */
      type = mytype;
      if (!tab->defCmod) {
          /* create default C model */
          IFnewUid(ckt, &uid, NULL, "C", UID_MODEL, NULL);
          IFC(newModel, (ckt, type, &(tab->defCmod), uid));
      }
      IFC(newInstance, (ckt, tab->defCmod, &fast, name));
      if (error1 == 1) {		/* was a c=val construction */
        val = INPevaluate(&line, &error1, 1);	/* [<val>] */
#ifdef TRACE
        printf ("In INP2C, C=val construction: val=%g\n", val);
#endif
      }
    }
    
    if (error1 == 0) {        /* Looks like a number */
      ptemp.rValue = val;
      GCA(INPpName, ("capacitance", &ptemp, ckt, type, fast));
    } 
    
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
      ptemp.rValue = leadval;
      GCA(INPpName, ("capacitance", &ptemp, ckt, type, fast));
    }

    return;
}
