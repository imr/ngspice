/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: Paolo Nenzi 2000
Remarks:  This code is based on a version written by Serban Popescu which
          accepted an optional parameter ac. I have adapted his code
	  to conform to INP standard. (PN)
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "inpdefs.h"
#include "inpmacs.h"
#include "fteext.h"
#include "inp.h"

void INP2R(void *ckt, INPtables * tab, card * current)
{
/* parse a resistor card */
/* Rname <node> <node> [<val>][<mname>][w=<val>][l=<val>][ac=<val>] */

    int mytype;			/* the type we determine resistors are */
    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *saveline;		/* ... just in case we need to go back... */
    char *name;			/* the resistor's name */
    char *model;		/* the name of the resistor's model */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    double val;			/* temp to held resistance */
    int error;			/* error code temporary */
    int error1;			/* secondary error code temporary */
    INPmodel *thismodel;	/* pointer to model structure describing our model */
    void *mdfast = NULL;	/* pointer to the actual model */
    void *fast;			/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model */

    mytype = INPtypelook("Resistor");
    if (mytype < 0) {
	LITERR("Device type Resistor not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    val = INPevaluate(&line, &error1, 1);
    /* either not a number -> model, or
     * follows a number, so must be a model name
     * -> MUST be a model name (or null)
     */

    saveline = line;		/* save then old pointer */

    INPgetTok(&line, &model, 1);

    if (*model) {
	/* token isn't null */
	if (INPlookMod(model)) {
	    /* If this is a valid model connect it */
	    INPinsert(&model, tab);
	    thismodel = (INPmodel *) NULL;
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
	    /* It is not a model */
	    line = saveline;	/* go back */
	    type = mytype;
	    if (!tab->defRmod) {	/* create default R model */
		IFnewUid(ckt, &uid, (IFuid) NULL, "R", UID_MODEL,
			 (void **) NULL);
		IFC(newModel, (ckt, type, &(tab->defRmod), uid));
	    }
	    mdfast = tab->defRmod;
	}
	IFC(newInstance, (ckt, mdfast, &fast, name));
    } else {
	/* The token is null and a default model will be created */
	type = mytype;
	if (!tab->defRmod) {
	    /* create default R model */
	    IFnewUid(ckt, &uid, (IFuid) NULL, "R", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defRmod), uid));
	}
	IFC(newInstance, (ckt, tab->defRmod, &fast, name));
    }

    if (error1 == 0) {		/* got a resistance above */
	ptemp.rValue = val;
	GCA(INPpName, ("resistance", &ptemp, ckt, type, fast))
    }

    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("resistance", &ptemp, ckt, type, fast));
    }
    return;
}
