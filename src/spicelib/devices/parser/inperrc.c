/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "inp.h"


char *INPerrCat(char *a, char *b)
{

    if (a != (char *) NULL) {
	if (b == (char *) NULL) {	/* a valid, b null, return a */
	    return (a);
	} else {		/* both valid  - hard work... */
	    register char *errtmp;
	    errtmp =
		(char *) MALLOC((strlen(a) + strlen(b) + 2) *
				sizeof(char));
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
