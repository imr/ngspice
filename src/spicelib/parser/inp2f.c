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

void INP2F(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* Fname <node> <node> <vname> <val> */

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
    IFuid uid;			/* uid of default model to be created */

    type = INPtypelook("CCCS");
    if (type < 0) {
	LITERR("Device type CCCS not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    if (!tab->defFmod) {
	/* create default F model */
	IFnewUid(ckt, &uid, NULL, "F", UID_MODEL, NULL);
	IFC(newModel, (ckt, type, &(tab->defFmod), uid));
    }
    
    /* call newInstance with macro IFC */    
    IFC(newInstance, (ckt, tab->defFmod, &fast, name));
    
    /* call bindNode with macro IFC */    
    IFC(bindNode, (ckt, fast, 1, node1));
    
    /* call bindNode with macro IFC */    
    IFC(bindNode, (ckt, fast, 2, node2));
    
    parm = INPgetValue(ckt, &line, IF_INSTANCE, tab);
    
    /* call INPpName with macro GCA */    
    GCA(INPpName, ("control", parm, ckt, type, fast));

    /* call INPdevParse with macro PARSECALL */    
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    
    if (waslead) {
	ptemp.rValue = leadval;
	
	/* call INPpName with macro GCA */	
	GCA(INPpName, ("gain", &ptemp, ckt, type, fast));
    }
}
