/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

void INP2D(CKTcircuit *ckt, INPtables * tab, card * current)
{

/* Dname <node> <node> <model> [<val>] [OFF] [IC=<val>] */

    int mytype;			/* the type we looked up */
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
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    GENmodel *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid of default model */

    mytype = INPtypelook("Diode");
    if (mytype < 0) {
	LITERR("Device type Diode not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetTok(&line, &model, 1);
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if ((mytype != thismodel->INPmodType)

#ifdef CIDER
                      && (thismodel->INPmodType != INPtypelook("NUMD"))
                      && (thismodel->INPmodType != INPtypelook("NUMD2"))
#endif
       ){
	    LITERR("incorrect model type");
	    return;
	}
	type = thismodel->INPmodType; /*HT 050903*/
	mdfast = (thismodel->INPmodfast);
    } else {
	type = mytype;
	if (!tab->defDmod) {
	    /* create default D model */
	    IFnewUid(ckt, &uid, NULL, "D", UID_MODEL,
		     NULL);
	    IFC(newModel, (ckt, type, &(tab->defDmod), uid));
	}
	mdfast = tab->defDmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {

#ifdef CIDER
    	if( type == INPtypelook("NUMD2") ) {
            LITERR(" error:  no unlabelled parameter permitted on NUMD2\n")
	} else {
#endif
	ptemp.rValue = leadval;
	GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
#ifdef CIDER    
  }
#endif
}
