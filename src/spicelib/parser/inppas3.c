/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: AlansFixes 
**********/

#include <config.h>
#include <ngspice.h>
#include <iferrmsg.h>
#include <ifsim.h>
#include <inpmacs.h>

#include "inppas3.h"

extern IFsimulator *ft_sim;


/* pass 3 - Read all nodeset and IC lines. All circuit nodes will have
 * been created by now, (except for internal device nodes), so any
 * nodeset or IC nodes which have to be created are flagged with a
 * warning.  */

void
INPpas3(void *ckt, card *data, INPtables *tab, void *task,
	IFparm *nodeParms, int numNodeParms)
{

    card *current;
    int error;			/* used by the macros defined above */
    char *line;			/* the part of the current line left
                                   to parse */
    char *name;			/* the node's name */
    char *token=NULL;		/* a token from the line */
    IFparm *prm;		/* pointer to parameter to search
                                   through array */
    IFvalue ptemp;		/* a value structure to package
                                   resistance into */
    int which;			/* which analysis we are performing */
    int length;			/* length of a name */
    void *node1;		/* the first node's node pointer */

#ifdef TRACE
    /* SDB debug statement */
    printf("In INPpas3 . . . \n");
#endif

    for(current = data; current != NULL; current = current->nextcard) {
	line = current->line;
	FREE(token)
	INPgetTok(&line,&token,1);

	if (strcmp(token,".nodeset")==0) {
	    which = -1;

	    for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
		if(strcmp(prm->keyword,"nodeset")==0) {
		    which = prm->id;
		    break;
		}
	    }

	    if(which == -1) {
		LITERR("nodeset unknown to simulator. \n")
		goto quit;
	    }

	    for(;;) {
		/* loop until we run out of data */
		INPgetTok(&line,&name,1);

		/* check to see if in the form V(xxx) and grab the xxx */
		if( *name == (char)NULL) break; /* end of line */
		length = strlen(name);
		if( (*name == 'V' || *(name) == 'v') && (length == 1)){
		    /* looks like V - must be V(xx) - get xx now*/
		    INPgetTok(&line,&name,1);
		    if (INPtermInsert(ckt,&name,tab,&node1)!=E_EXISTS)
			fprintf(stderr,
				"Warning : Nodeset on non-existant node - %s\n", name);
		    ptemp.rValue = INPevaluate(&line,&error,1);
		    IFC(setNodeParm,(ckt,node1,which,&ptemp,(IFvalue*)NULL));
		    continue;
		}
		LITERR(" Error: .nodeset syntax error.\n")
		    break;
	    }
	} else if ((strcmp(token,".ic") == 0)) {
	    /* .ic */
	    which = -1;
	    for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
		if(strcmp(prm->keyword,"ic")==0) {
		    which = prm->id;
		    break;
		}
	    }

	    if(which==-1) {
		LITERR("ic unknown to simulator. \n")
		goto quit;
	    }

	    for(;;) {
		/* loop until we run out of data */
		INPgetTok(&line,&name,1);
		/* check to see if in the form V(xxx) and grab the xxx */
		if( *name == 0) break; /* end of line */
		length = strlen(name);
		if( (*name == 'V' || *(name) == 'v') && (length == 1)){
		    /* looks like V - must be V(xx) - get xx now*/
		    INPgetTok(&line,&name,1);
		    if (INPtermInsert(ckt,&name,tab,&node1)!=E_EXISTS)
			fprintf(stderr,
				"Warning : IC on non-existant node - %s\n", name);
		    ptemp.rValue = INPevaluate(&line,&error,1);
		    IFC(setNodeParm,(ckt,node1,which,&ptemp,(IFvalue*)NULL))
			continue;
		}
		LITERR(" Error: .ic syntax error.\n")
		    break;
	    }
	}
    }
quit:
    FREE(token);
   return;
}

