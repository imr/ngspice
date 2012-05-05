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
#include "inp.h"

void INP2U(CKTcircuit *ckt, INPtables * tab, card * current)
{
#if ADMS >= 3

    /* Uname <node> <node> ... <model> [param1=<val1>] [param1=<val2>] ... */

    char *line;                 /* the part of the current line left to parse */
    char *iname;                 /* the instance's name */
    char *name;                 /* the resistor's name */
    int nsize;                /* number of nodes */
    int i;
    CKTnode **node;             /* the first node's node pointer */
    int error;                  /* error code temporary */
    GENinstance *fast;          /* pointer to the actual instance */
    int waslead;                /* flag to indicate that funny unlabeled number was found */
    double leadval;             /* actual value of unlabeled number */

#ifdef TRACE
    printf("INP2U: Parsing '%s'\n", current->line);
#endif

    nsize = 0;
    node=NULL;
    line = current->line;

    INPgetTok(&line, &iname, 1);
    INPinsert(&iname, tab);

    INPgetNetTok(&line, &name, 1);
    while(!INPlookMod(name) && (*line != '\0'))
    {
#ifdef TRACE
      printf("INP2U: found node %s\n",name);
#endif
      nsize++;
      node=TREALLOC(CKTnode*,node,nsize);
      INPtermInsert(ckt, &name, tab, &node[nsize-1]);
      INPgetNetTok(&line, &name, 1);
    }

    if (INPlookMod(name)) {
        INPmodel *thismodel;        /* pointer to model description for user's model */
        thismodel = NULL;
        INPinsert(&name, tab);
#ifdef TRACE
        printf("INP2U: found dynamic model %s\n",name);
#endif
        current->error = INPgetMod(ckt, name, &thismodel, tab);
        if (thismodel == NULL) {
      	   fprintf(stderr, "%s\nPlease check model, level or number of terminals!\n", current->error);
      	   controlled_exit(EXIT_BAD);
        }
        else
        {
          IFC(newInstance, (ckt, thismodel->INPmodfast, &fast, iname));
          for(i=0;i<nsize;i++)
          {
            IFC(bindNode, (ckt, fast, i+1, node[i]));
          }
          PARSECALL((&line, ckt, thismodel->INPmodType, fast, &leadval, &waslead, tab));
#ifdef TRACE
          printf("INP2U: Looking up model done\n");
#endif
        }
    } else {
      	   fprintf(stderr, "Unable to find definition of model %s\n", name);
      	   controlled_exit(EXIT_BAD);
    }

#else
    /* Uname <node> <node> <model> [l=<val>] [n=<val>] */

    int mytype;			/* the type my lookup says URC is */
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
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    char *model;		/* name of the model */
    INPmodel *thismodel;	/* pointer to our model descriptor */
    GENmodel *mdfast;		/* pointer to the actual model */
    IFuid uid;			/* uid for default model */

    mytype = INPtypelook("URC");
    if (mytype < 0) {
	LITERR("Device type URC not supported by this binary\n");
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
    INPinsert(&model, tab);
    current->error = INPgetMod(ckt, model, &thismodel, tab);
    if (thismodel != NULL) {
	if (mytype != thismodel->INPmodType) {
	    LITERR("incorrect model type");
	    return;
	}
	type = mytype;
	mdfast = (thismodel->INPmodfast);
    } else {
	type = mytype;
	if (!tab->defUmod) {
	    /* create deafult U model */
	    IFnewUid(ckt, &uid, NULL, "U", UID_MODEL,
		     NULL);
	    IFC(newModel, (ckt, type, &(tab->defUmod), uid));
	}
	mdfast = tab->defUmod;
    }
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
#endif
}
