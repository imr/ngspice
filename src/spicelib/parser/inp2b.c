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

void INP2B(void *ckt, INPtables * tab, card * current)
{

    /* Bname <node> <node> [V=expr] [I=expr] */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    int error;			/* error code temporary */
    void *fast;			/* pointer to the actual instance */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model name */

    /* Arbitrary source. */
    type = INPtypelook("ASRC");
    if (type < 0) {
	LITERR("Device type Asource not supported by this binary\n");
	return;
    }

    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);

    INPgetNetTok(&line, &nname1, 1);
    error = INPtermInsert(ckt, &nname1, tab, &node1);

    INPgetNetTok(&line, &nname2, 1);
    error = INPtermInsert(ckt, &nname2, tab, &node2);

    if (!tab->defBmod) {
	/* create default B model */
	IFnewUid(ckt, &uid, (IFuid) NULL, "B", UID_MODEL, (void **) NULL);
	IFC(newModel, (ckt, type, &(tab->defBmod), uid));
    }
    IFC(newInstance, (ckt, tab->defBmod, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
}
