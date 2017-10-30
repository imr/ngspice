/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* INPlist(file,deck,type)
     *  provide an input listing on the specified file of the given
     *  card deck.  The listing should be of either PHYSICAL or LOGICAL
     *  lines as specified by the type parameter.
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"

extern void INPlist(FILE *file, struct card *deck, int type);  /* nowhere used function */

void INPlist(FILE * file, struct card *deck, int type)
{

    struct card *here;
    struct card *there;

    if (type == LOGICAL) {
	for (here = deck; here != NULL; here = here->nextcard) {
	    fprintf(file, "%6d : %s\n", here->linenum, here->line);
	    if (here->error != NULL) {
		fprintf(file, "%s", here->error);
	    }
	}
    } else if (type == PHYSICAL) {
	for (here = deck; here != NULL; here = here->nextcard) {
	    if (here->actualLine == NULL) {
		fprintf(file, "%6d : %s\n", here->linenum, here->line);
		if (here->error != NULL) {
		    fprintf(file, "%s", here->error);
		}
	    } else {
		for (there = here->actualLine; there != NULL;
		     there = there->nextcard) {
		    fprintf(file, "%6d : %s\n", there->linenum,
			    there->line);
		    if (there->error != NULL) {
			fprintf(file, "%s", there->error);
		    }
		}
	    }
	}
    }
}
