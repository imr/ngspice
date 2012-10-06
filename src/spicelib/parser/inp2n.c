/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#include "ngspice/ngspice.h"

#ifdef NDEV

#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "../devices/ndev/ndevdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inp.h"

void INP2N(CKTcircuit *ckt, INPtables * tab, card * current)
{
/* parse a numerical device  card */
/* Nname <node> <node> [<node> ...] [<mname>] */
/* The NUMD should have a private .model card */

    int mytype;			/* the type we determine NDEV are */
    int type = 0;		/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *saveline;		/* ... just in case we need to go back... */
    char *name;			/* the NDEV's name */
    char *model;		/* the name of the NDEV's model */
    
    int  term;                  /* the number of node */
    char *nnamex;               /* serve as a temporary name */ 
    char *nname[7];		/* the array of CKT node's name */
    char *bname[7];		/* the array of NDEV electrode's name */
    CKTnode *node[7];		/* the array of CKT node's node pointer */
    
    int error;			/* error code temporary */
    int i;            
    INPmodel *thismodel;	/* pointer to model structure describing our model */
    GENmodel *mdfast = NULL;	/* pointer to the actual model */
    GENinstance *fast;		/* pointer to the actual instance */
    NDEVinstance *pinst;
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */

    mytype = INPtypelook("NDEV");
    if (mytype < 0) {
	LITERR("Device type NDEV not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    
    /* get the node number here */
    saveline=line;
    term = 0;
    do {
      INPgetNetTok(&line, &nnamex, 1);
      term++;
    }while(*nnamex);
    line=saveline;
    term=(term-2)/2;
    if (term > 7) {
	LITERR("Numerical device has too much nodes, the limitation is 7\n");
	return;
    }
    for(i=0;i<term;i++) {   
      INPgetNetTok(&line, &nname[i], 1);
      INPgetNetTok(&line, &bname[i], 1);
      INPtermInsert(ckt, &nname[i], tab, &node[i]);
    }  

    saveline = line;		/* save then old pointer */

    INPgetTok(&line, &model, 1);
          
    if (*model) {
	/* token isn't null */
	if (INPlookMod(model)) {
	    /* If this is a valid model connect it */
	    INPinsert(&model, tab);
	    thismodel = NULL;
	    current->error = INPgetMod(ckt, model, &thismodel, tab);
	    if (thismodel != NULL) {
		if (mytype != thismodel->INPmodType) {
		    LITERR("incorrect model type");
		    return;
		}
		mdfast = thismodel->INPmodfast;
		type = thismodel->INPmodType;
	    }
	} else {
	    LITERR("Numerical device should always have a model card\n");
	    return;
	}
	IFC(newInstance, (ckt, mdfast, &fast, name));
    } else {
	LITERR("Numerical device should always have a model card\n");
	return;
    }

    for(i=0;i<term;i++) {   
    	IFC(bindNode, (ckt, fast, i+1, node[i]));
    }	
    /* save acture terminal number to instance */
    pinst = (NDEVinstance *)fast;
    pinst->term = term;
    for(i=0;i<term;i++) {   
       pinst->bname[i]=bname[i];
       pinst->node[i]=node[i];
    }  
    
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	LITERR("The numerical device was lead berfor.\n");
	return;
    }
    
    
    return;
}
#else

int Dummy1;

#endif
