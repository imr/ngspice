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

void INP2J(void *ckt, INPtables * tab, card * current)
{

    /* Jname <node> <node> <node> <model> [<val>] [OFF] [IC=<val>,<val>] */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    char *nname3;		/* the third node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    void *node3;		/* the third node's node pointer */
    int error;			/* error code temporary */
    void *fast;			/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    void *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid of default model */

    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetNetTok(&line, &nname3, 1);
    INPtermInsert(ckt, &nname3, tab, &node3);
    INPgetTok(&line, &model, 1);
    INPinsert(&model, tab);
    thismodel = (INPmodel *) NULL;
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if (thismodel->INPmodType != INPtypelook("JFET")
	    && thismodel->INPmodType != INPtypelook("JFET2")
	    ) {
	    LITERR("incorrect model type");
	    return;
	}
	type = thismodel->INPmodType;
	mdfast = (thismodel->INPmodfast);
    } else {
	type = INPtypelook("JFET");
	if (type < 0) {
	    LITERR("Device type JFET not supported by this binary\n");
	    return;
	}
	if (!tab->defJmod) {
	    /* create default J model */
	    IFnewUid(ckt, &uid, (IFuid) NULL, "J", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defJmod), uid));
	}
	mdfast = tab->defJmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
}
