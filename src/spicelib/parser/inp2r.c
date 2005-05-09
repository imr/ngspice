/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: Paolo Nenzi 2000
Remarks:  This code is based on a version written by Serban Popescu which
          accepted an optional parameter ac. I have adapted his code
	  to conform to INP standard. (PN)
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "inpdefs.h"
#include "inpmacs.h"
#include "fteext.h"
#include "inp.h"

/* undefine to add tracing to this file */
/*#define TRACE*/

void INP2R(void *ckt, INPtables * tab, card * current)
{
/* parse a resistor card */
/* Rname <node> <node> [<val>][<mname>][w=<val>][l=<val>][ac=<val>] */

    int mytype;			/* the type we determine resistors are */
    int type = 0;		/* the type the model says it is */
    char *line;			/* the part of the current line left to parse */
    char *saveline;		/* ... just in case we need to go back... */
    char *name;			/* the resistor's name */
    char *model;		/* the name of the resistor's model */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    double val;			/* temp to held resistance */
    int error;			/* error code temporary */
    int error1;			/* secondary error code temporary */
    INPmodel *thismodel;	/* pointer to model structure describing our model */
    void *mdfast = NULL;	/* pointer to the actual model */
    void *fast;			/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int waslead;		/* flag to indicate that funny unlabeled number was found */
    double leadval;		/* actual value of unlabeled number */
    IFuid uid;			/* uid for default model */

    char *s,*t;/* Temporary buffer and pointer for translation */
    int i;
	
#ifdef TRACE
    printf("In INP2R()\n");
    printf("  Current line: %s\n",current->line);
#endif

    mytype = INPtypelook("Resistor");
    if (mytype < 0) {
	LITERR("Device type Resistor not supported by this binary\n");
	return;
    }
    line = current->line;
    INPgetTok(&line, &name, 1);			/* Rname */
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);		/* <node> */
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);		/* <node> */
    INPtermInsert(ckt, &nname2, tab, &node2);
    val = INPevaluate(&line, &error1, 1);	/* [<val>] */
    /* either not a number -> model, or
     * follows a number, so must be a model name
     * -> MUST be a model name (or null)
     */
    
    saveline = line;		/* save then old pointer */
    
#ifdef TRACE
    printf("Begining tc=xxx yyyy search and translation in '%s'\n", line);
#endif /* TRACE */    
    
    /* This routine translates "tc=xxx yyy" to "tc=xxx tc2=yyy".
       This is a re-write of the routine originally proposed by Hitoshi tanaka.
       In my version we simply look for the first occurence of 'tc' followed
       by '=' followed by two numbers. If we find it then we splice in "tc2=".
       sjb - 2005-05-09 */
    for(s=line; *s; s++) { /* scan the remainder of the line */
	
	/* reject anything other than "tc" */
	if(*s!='t') continue;
	s++;
	if(*s!='c') continue;
	s++;
	
	/* skip any white space */
	while(isspace(*s)) s++;
	      
	/* reject if not '=' */
	if(*s!='=') continue;
	s++;
	
	/* skip any white space */
	while(isspace(*s)) s++;
	
	/* if we now have +, - or a decimal digit then assume we have a number,
	   otherwise reject */
	if((*s!='+') && (*s!='-') && !isdigit(*s)) continue;
	s++;
	
	/* look for next white space or null */
	while(!isspace(*s) && *s) s++;
	
	/* reject whole line if null (i.e not white space) */
	if(*s==0) break;
	s++;
	
	/* remember this location in the line.
	   Note that just before this location is a white space character. */
	t = s;
	
	/* skip any additional white space */
	while(isspace(*s)) s++;
	
	/* if we now have +, - or a decimal digit then assume we have the 
	    second number, otherwise reject */
	if((*s!='+') && (*s!='-') && !isdigit(*s)) continue;
	
	/* if we get this far we have met all are criterea,
	   so now we splice in a "tc2=" at the location remembered above. */
	
	/* first alocate memory for the new longer line */
	i = strlen(current->line);  /* length of existing line */	
	line = tmalloc(i + 4 + 1);  /* alocate enough for "tc2=" & terminating NULL */
	if(line == NULL) {
	    /* failed to allocate memory so we recover rather crudely
	       by rejecting the translation */
	    line = saveline;
	    break;
	}
	
	/* copy first part of line */
	i -= strlen(t);
	strncpy(line,current->line,i);
	line[i]=0;  /* terminate */
	
	/* add "tc2=" */
	strcat(line, "tc2=");
	
	/* add rest of line */
	strcat(line, t);
	
	/* calculate our saveline position in the new line */
	saveline = line + (saveline - current->line);
	
	/* replace old line with new */
	tfree(current->line);
	current->line = line;
	line = saveline;
    }
    
#ifdef TRACE
    printf("(Translated) Res line: %s\n",current->line);
#endif

    INPgetTok(&line, &model, 1);

    if (*model) {
	/* token isn't null */
	if (INPlookMod(model)) {
	    /* If this is a valid model connect it */
	    INPinsert(&model, tab);
	    thismodel = (INPmodel *) NULL;
	    current->error = INPgetMod(ckt, model, &thismodel, tab);
	    if (thismodel != NULL) {
		if (mytype != thismodel->INPmodType) {
		    LITERR("incorrect model type for resistor");
		    return;
		}
		mdfast = thismodel->INPmodfast;
		type = thismodel->INPmodType;
	    }
	} else {
	    tfree(model);
	    /* It is not a model */
	    line = saveline;	/* go back */
	    type = mytype;
	    if (!tab->defRmod) {	/* create default R model */
		IFnewUid(ckt, &uid, (IFuid) NULL, "R", UID_MODEL,
			 (void **) NULL);
		IFC(newModel, (ckt, type, &(tab->defRmod), uid));
	    }
	    mdfast = tab->defRmod;
	}
	IFC(newInstance, (ckt, mdfast, &fast, name));
    } else {
	tfree(model);
	/* The token is null and a default model will be created */
	type = mytype;
	if (!tab->defRmod) {
	    /* create default R model */
	    IFnewUid(ckt, &uid, (IFuid) NULL, "R", UID_MODEL,
		     (void **) NULL);
	    IFC(newModel, (ckt, type, &(tab->defRmod), uid));
	}
	IFC(newInstance, (ckt, tab->defRmod, &fast, name));
    }

    if (error1 == 0) {		/* got a resistance above */
	ptemp.rValue = val;
	GCA(INPpName, ("resistance", &ptemp, ckt, type, fast))
    }

    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
	ptemp.rValue = leadval;
	GCA(INPpName, ("resistance", &ptemp, ckt, type, fast));
    }

    return;
}
