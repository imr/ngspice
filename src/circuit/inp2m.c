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

void INP2M(void *ckt, INPtables * tab, card * current)
{

    /* Mname <node> <node> <node> <node> <model> [L=<val>]
     *       [W=<val>] [AD=<val>] [AS=<val>] [PD=<val>]
     *       [PS=<val>] [NRD=<val>] [NRS=<val>] [OFF] 
     *       [IC=<val>,<val>,<val>] 
     */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    char *nname3;		/* the third node's name */
    char *nname4;		/* the fourth node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    void *node3;		/* the third node's node pointer */
    void *node4;		/* the fourth node's node pointer */
    int error;			/* error code temporary */
    void *fast;			/* pointer to the actual instance */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    void *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid for default model */

    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetTok(&line, &nname3, 1);
    INPtermInsert(ckt, &nname3, tab, &node3);
    INPgetTok(&line, &nname4, 1);
    INPtermInsert(ckt, &nname4, tab, &node4);
    INPgetTok(&line, &model, 1);
    INPinsert(&model, tab);
    thismodel = (INPmodel *) NULL;
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if (thismodel->INPmodType != INPtypelook("Mos1")
	    && thismodel->INPmodType != INPtypelook("Mos2")
	    && thismodel->INPmodType != INPtypelook("Mos3")
	    && thismodel->INPmodType != INPtypelook("Mos5")
	    && thismodel->INPmodType != INPtypelook("Mos6")
	    && thismodel->INPmodType != INPtypelook("Mos8")
	    && thismodel->INPmodType != INPtypelook("BSIM1")
	    && thismodel->INPmodType != INPtypelook("BSIM2")
	    && thismodel->INPmodType != INPtypelook("BSIM3")
	    && thismodel->INPmodType != INPtypelook("BSIM4")
	    && thismodel->INPmodType != INPtypelook("BSIM3V1")
	    && thismodel->INPmodType != INPtypelook("BSIM3V2")
	    ) {
	    LITERR("incorrect model type");
	    return;
	}
	type = thismodel->INPmodType;
	mdfast = (thismodel->INPmodfast);
    } else {
	type = INPtypelook("Mos1");
	if (type < 0) {
	    LITERR("Device type MOS1 not supported by this binary\n");
	    return;
	}
	if (!tab->defMmod) {
	    /* create default M model */
	    IFnewUid(ckt, &uid, (IFuid) NULL, "M", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defMmod), uid));
	}
	mdfast = tab->defMmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    IFC(bindNode, (ckt, fast, 4, node4));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	LITERR(" error:  no unlabeled parameter permitted on mosfet\n");
    }
}
