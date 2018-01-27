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

void INP2H(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* Hname <node> <node> <vname> <val> */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* pointer to a value structure for functions which return one */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model to be created */

    type = INPtypelook("CCVS");
    if (type < 0) {
	LITERR("Device type CCVS not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    if (!tab->defHmod) {
	/* create default H model */
	IFnewUid(ckt, &uid, NULL, "H", UID_MODEL, NULL);
	IFC(newModel, (ckt, type, &(tab->defHmod), uid));
    }
    IFC(newInstance, (ckt, tab->defHmod, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    parm = INPgetValue(ckt, &line, IF_INSTANCE, tab);
    GCA(INPpName, ("control", parm, ckt, type, fast));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("gain", &ptemp, ckt, type, fast));
    }
}
