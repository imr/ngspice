/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

/* Added code from macspice3f4 HFET1&2 and MESA model 
   Original note:
     Added device calls for Mesfet models  and HFET models
   provided by Trond Ytterdal  as of Nov 98 
*/


#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

void INP2Z(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

    /* Zname <node> <node> <node> <model> [<val>] [OFF] [IC=<val>,<val>] */

    int type;			/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    char *nname3;		/* the third node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    CKTnode *node3;		/* the third node's node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* the name of the model */
    INPmodel *thismodel;	/* pointer to model description for user's model */
    GENmodel *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid for default model */

   
    line = current->line;
    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetNetTok(&line, &nname3, 1);
    INPtermInsert(ckt, &nname3, tab, &node3);
    INPgetNetTok(&line, &model, 1);
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
		if (   thismodel->INPmodType != INPtypelook("MES") 
    		    && thismodel->INPmodType != INPtypelook("MESA")
    		    && thismodel->INPmodType != INPtypelook("HFET1")
    		    && thismodel->INPmodType != INPtypelook("HFET2")) 
    	{
            LITERR("incorrect model type");
            return;
        }
	
	
	type = thismodel->INPmodType;
	mdfast = (thismodel->INPmodfast);
    } else {
    	
     type = INPtypelook("MES");
	if (type < 0 ) {
			LITERR("Device type MES not supported by this binary\n");
           	return;
        }    
		
	if (!tab->defZmod) {
	    /* create default Z model */
	    IFnewUid(ckt, &uid, NULL, "Z", UID_MODEL, NULL);
	    IFC(newModel, (ckt, type, &(tab->defZmod), uid));
	}
	mdfast = tab->defZmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
  /* use type - not thismodel->INPmodType as it might not exist! */
    /* FIXME: Why do we need checking for type here? */
    if ( (waslead) && ( type /*thismodel->INPmodType*/ != INPtypelook("MES") ) ) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
}
