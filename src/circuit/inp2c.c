/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "inpdefs.h"
#include "inpmacs.h"
#include "fteext.h"
#include "inp.h"

void INP2C(void *ckt, INPtables * tab, card * current)
{

/* parse a capacitor card */
/* Cname <node> <node> <val> [IC=<val>] */

    int mytype;			/* the type we determine resistors are */
    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *model;		/* the name of the resistor's model */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    double val;			/* temp to held resistance */
    int error;			/* error code temporary */
    INPmodel *thismodel;	/* pointer to model structure describing our model */
    void *mdfast;		/* pointer to the actual model */
    void *fast;			/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default cap model */

    mytype = INPtypelook("Capacitor");
    if (mytype < 0) {
	LITERR("Device type Capacitor not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    val = INPevaluate(&line, &error, 1);
    if (error == 0) {		/* Looks like a number */
	type = mytype;
	ptemp.rValue = val;
	if (!tab->defCmod) {
	    IFnewUid(ckt, &uid, (IFuid) NULL, "C", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defCmod), uid))
	}
	IFC(newInstance, (ckt, tab->defCmod, &fast, name))
	    GCA(INPpName, ("capacitance", &ptemp, ckt, type, fast))
    } else {			/* looks like character strings */
	INPgetTok(&line, &model, 1);
	INPinsert(&model, tab);
	thismodel = (INPmodel *) NULL;
	current->error = INPgetMod(ckt, model, &thismodel, tab);
	if (thismodel != NULL) {
	    if (mytype != thismodel->INPmodType) {
		LITERR("incorrect model type");
		return;
	    }
	    type = mytype;
	    mdfast = thismodel->INPmodfast;
	} else {
	    type = mytype;
	    if (!tab->defCmod) {
		IFnewUid(ckt, &uid, (IFuid) NULL, "C", UID_MODEL,
			 (void **) NULL);
		IFC(newModel, (ckt, type, &(tab->defCmod), uid))
	    }
	    mdfast = tab->defCmod;
	}
	IFC(newInstance, (ckt, mdfast, &fast, name))
    }
    IFC(bindNode, (ckt, fast, 1, node1))
	IFC(bindNode, (ckt, fast, 2, node2))
	PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab))
	if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("capacitance", &ptemp, ckt, type, fast))
    }
}
