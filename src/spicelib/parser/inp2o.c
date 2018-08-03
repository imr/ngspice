/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"


void INP2O(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

    /* Oname <node> <node> <node> <node> [IC=<val>,<val>,<val>,<val>] */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    char *nname3;		/* the third node's name */
    char *nname4;		/* the fourth node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    CKTnode *node3;		/* the third node's node pointer */
    CKTnode *node4;		/* the fourth node's node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    GENmodel *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid for default model */


    type = INPtypelook("LTRA");
    if (type < 0) {
	LITERR("Device type LossyXmissionLine not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetNetTok(&line, &nname3, 1);
    INPtermInsert(ckt, &nname3, tab, &node3);
    INPgetNetTok(&line, &nname4, 1);
    INPtermInsert(ckt, &nname4, tab, &node4);
    INPgetNetTok(&line, &model, 1);
    if (INPlookMod(model)) {
	/* do nothing for now */
	/* no action required */
    } else {
	/*
	   nname4 = model;
	   INPtermInsert(ckt,&nname4,tab,&node4);
	   INPgetNetTok(&line, &model, 1);
	 */
    }
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if (type != thismodel->INPmodType) {
	    LITERR("incorrect model type");
	    return;
	}
	mdfast = (thismodel->INPmodfast);
    } else {
	if (!tab->defOmod) {
	    /* create default O model */
	    IFnewUid(ckt, &uid, NULL, "O", UID_MODEL, NULL);
	    IFC(newModel, (ckt, type, &(tab->defOmod), uid));
	}
	mdfast = tab->defOmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    IFC(bindNode, (ckt, fast, 4, node4));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
}
