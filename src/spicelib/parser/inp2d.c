/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

void INP2D(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* Dname <node> <node> [<temp>] <model> [<val>] [OFF] [IC=<val>] */

    int mytype;         /* the type we looked up */
    int type;           /* the type the model says it is */
    char *line;         /* the part of the current line left to parse */
    char *name;         /* the resistor's name */
    const int max_i = 3;/* the maximum node count */
    CKTnode *node[3];   
    int error;          /* error code temporary */
    int numnodes;       /* flag indicating 4 or 5 nodes */
    GENinstance *fast;  /* pointer to the actual instance */
    IFvalue ptemp;      /* a value structure to package resistance into */
    int waslead;        /* flag to indicate that funny unlabeled number was found */
    double leadval;     /* actual value of unlabeled number */
    INPmodel *thismodel;/* pointer to model description for user's model */
    GENmodel *mdfast;   /* pointer to the actual model */
    IFuid uid;          /* uid of default model */
    int i;

    mytype = INPtypelook("Diode");
    if (mytype < 0) {
    LITERR("Device type Diode not supported by this binary\n");
    return;
    }
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);

    for (i = 0; ; i++) {
        char *token;
        INPgetNetTok(&line, &token, 1);
        if (i >= 2 && INPlookMod(token)) {
            INPinsert(&token, tab);
            txfree(INPgetMod(ckt, token, &thismodel, tab));
            if (!thismodel) {
                LITERR ("Unable to find definition of given model");
                return;
            }
            break;
        }
        if (i >= max_i) {
            LITERR ("could not find a valid modelname");
            return;
        }
        INPtermInsert(ckt, &token, tab, &node[i]);
    }

    if (i > max_i) {
        LITERR("Too many nodes for this model type");
        return;
    }

    numnodes = i;

    if (thismodel != NULL) {
        if ((mytype != thismodel->INPmodType)
#ifdef CIDER
            && (thismodel->INPmodType != INPtypelook("NUMD"))
            && (thismodel->INPmodType != INPtypelook("NUMD2"))
#endif
        ) {
            LITERR("incorrect model type");
            return;
        }
        type = thismodel->INPmodType; /*HT 050903*/
        mdfast = (thismodel->INPmodfast);
    } else {
        type = mytype;
        if (!tab->defDmod) {
            /* create default D model */
            IFnewUid(ckt, &uid, NULL, "D", UID_MODEL, NULL);
            IFC(newModel, (ckt, type, &(tab->defDmod), uid));
        }
        mdfast = tab->defDmod;
    }

    IFC(newInstance, (ckt, mdfast, &fast, name));
    for (i = 0; i < max_i; i++)
        if (i < numnodes)
            IFC (bindNode, (ckt, fast, i + 1, node[i]));
        else if (thismodel->INPmodType != INPtypelook("NUMD")
            && (thismodel->INPmodType != INPtypelook("NUMD2")))
            GENnode(fast)[i] = -1;

    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
#ifdef CIDER
        if( type == INPtypelook("NUMD2") ) {
            LITERR(" error:  no unlabelled parameter permitted on NUMD2\n");
    } else {
#endif
        ptemp.rValue = leadval;
        GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
#ifdef CIDER    
    }
#endif
}
