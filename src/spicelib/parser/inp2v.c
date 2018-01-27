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

void INP2V(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

    /* Vname <node> <node> [ [DC] <val>] [AC [<val> [<val> ] ] ]
     *       [<tran function>] */

    static int type = -1;	/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model */

    if (type < 0) {
        if ((type = INPtypelook("Vsource")) < 0) {
            LITERR("Device type Vsource not supported by this binary\n");
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
    if (!tab->defVmod) {
	/* create default V model */
	IFnewUid(ckt, &uid, NULL, "V", UID_MODEL, NULL);
	IFC(newModel, (ckt, type, &(tab->defVmod), uid));
    }
    IFC(newInstance, (ckt, tab->defVmod, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("dc", &ptemp, ckt, type, fast));
    }
}
