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

void INP2F(void *ckt, INPtables * tab, card * current)
{

/* Fname <node> <node> <vname> <val> */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    int error;			/* error code temporary */
    void *fast;			/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value structure to pick things up into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model */

    type = INPtypelook("CCCS");
    if (type < 0) {
	LITERR("Device type CCCS not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    if (!tab->defFmod) {
	/* create default F model */
	IFnewUid(ckt, &uid, (IFuid) NULL, "F", UID_MODEL, (void **) NULL);
	IFC(newModel, (ckt, type, &(tab->defFmod), uid))
    }
    IFC(newInstance, (ckt, tab->defFmod, &fast, name))
	IFC(bindNode, (ckt, fast, 1, node1))
	IFC(bindNode, (ckt, fast, 2, node2))
	parm = INPgetValue(ckt, &line, IF_INSTANCE, tab);
    GCA(INPpName, ("control", parm, ckt, type, fast))
	PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab))
	if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("gain", &ptemp, ckt, type, fast))
    }
}
