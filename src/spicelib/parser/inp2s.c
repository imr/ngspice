/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

void INP2S(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

    /* Sname <node> <node> <node> <node> [<modname>] [IC] */
    /*  VOLTAGE CONTROLLED SWITCH */

    int mytype;			/* the type we determine resistors are */
    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *model;		/* the name of the resistor's model */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    char *nname3;		/* the third node's name */
    char *nname4;		/* the fourth node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    CKTnode *node3;		/* the third node's node pointer */
    CKTnode *node4;		/* the fourth node's node pointer */
    int error;			/* error code temporary */
    INPmodel *thismodel;	/* pointer to model structure describing our model */
    GENmodel *mdfast;		/* pointer to the actual model */
    GENinstance *fast;		/* pointer to the actual instance */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid of default model */

    mytype = INPtypelook("Switch");
    if (mytype < 0) {
	LITERR("Device type Switch not supported by this binary\n");
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
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if (mytype != thismodel->INPmodType) {
	    LITERR("incorrect model type");
	    return;
	}
	type = mytype;
	mdfast = (thismodel->INPmodfast);
    } else {
	type = mytype;
	if (!tab->defSmod) {
	    /* create deafult S model */
	    IFnewUid(ckt, &uid, NULL, "S", UID_MODEL, NULL);
	    IFC(newModel, (ckt, type, &(tab->defSmod), uid));
	}
	mdfast = tab->defSmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    IFC(bindNode, (ckt, fast, 4, node4));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	/* ignore a number */
    }
}
