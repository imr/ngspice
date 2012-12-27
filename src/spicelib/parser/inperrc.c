/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"


char *INPerrCat(char *a, char *b)
{

    if (a != NULL) {
	if (b == NULL) {	/* a valid, b null, return a */
	    return (a);
	} else {		/* both valid  - hard work... */
	    register char *errtmp;
	    errtmp =
		TMALLOC(char, strlen(a) + strlen(b) + 2);
	    (void) strcpy(errtmp, a);
	    (void) strcat(errtmp, "\n");
	    (void) strcat(errtmp, b);
	    FREE(a);
	    FREE(b);
	    return (errtmp);
	}
    } else {			/* a null, so return b */
	return (b);
    }
}
