/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "inpdefs.h"
#include "inpmacs.h"
#include "fteext.h"
#include "inp.h"

void INP2Q(void *ckt, INPtables * tab, card * current, void *gnode)
{

    /* Qname <node> <node> <node> [<node>] <model> [<val>] [OFF]
     *       [IC=<val>,<val>] */

    int mytype;			/* the type we looked up */
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
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    void *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid of default model */

    mytype = INPtypelook("BJT");
    if (mytype < 0) {
	LITERR("Device type BJT not supported by this binary\n");
	return;
    }
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
    if (INPlookMod(model)) {
	/* do nothing for now */
	node4 = gnode;
	/* no action required */
    } else {
	nname4 = model;
	INPtermInsert(ckt, &nname4, tab, &node4);
	INPgetTok(&line, &model, 1);
    }
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if((thismodel->INPmodType != INPtypelook("BJT"))
           && (thismodel->INPmodType != INPtypelook("BJT2"))
           && (thismodel->INPmodType != INPtypelook("VBIC"))
#ifdef CIDER
           && (thismodel->INPmodType != INPtypelook("NBJT"))
           && (thismodel->INPmodType != INPtypelook("NBJT2"))
#endif
         ) {
            LITERR("incorrect model type")
            return;
        }
        type = (thismodel->INPmodType);
        mdfast = (thismodel->INPmodfast);    
    } else {
	type = mytype;
	if (!tab->defQmod) {
	    /* create default Q model */
	    IFnewUid(ckt, &uid, (IFuid) NULL, "Q", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defQmod), uid));
	}
	mdfast = tab->defQmod;
    }
    
#ifdef TRACE
    /* ---  SDB debug statement --- */
    printf ("In INP2Q, just about to dive into newInstance\n");
#endif
    
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    IFC(bindNode, (ckt, fast, 4, node4));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
#ifdef CIDER
    if( type == INPtypelook("NBJT2") ) {
            LITERR(" error:  no unlabelled parameter permitted on NBJT2\n")
 	} else {
#endif
	ptemp.rValue = leadval;
	GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
#ifdef CIDER
   }
#endif   
}
