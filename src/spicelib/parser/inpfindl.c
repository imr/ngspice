/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi - Now we can use a two digits level code -
**********/

    /* INPfindLev(line,level)
     *      find the 'level' parameter on the given line and return its
     *      value (1,2,..,99 for now, 1 default)
     * 
     */

#include "ngspice.h"
#include <stdio.h>
#include <string.h>
#include "inpdefs.h"
#include "inp.h"

char *INPfindLev(char *line, int *level)
{
    char *where;

    /*
     *where = line;
     */

    where = strstr(line, "level");

    if (where != NULL) {	/* found a level keyword on the line */

	where += 5;		/* skip the level keyword */
	while ((*where == ' ') || (*where == '\t') || (*where == '=') ||
	       (*where == ',') || (*where == '(') || (*where == ')') ||
	       (*where == '+')) {	/* legal white space - ignore */
	    where++;
	}

	/* now the magic number */
	sscanf(where, "%2d", level);	/* We get the level number */
	if (*level < 0) {
	    *level = 1;
	    printf("Illegal value for level.\n");
	    printf("Level must be >0 (Setting level to 1)\n");
	    return (INPmkTemp
		    (" illegal (negative) argument to level parameter - level=1 assumed"));
	}

	if (*level > 99) {	/* Limit to change in the future */
	    *level = 1;
	    printf("Illegal value for level.\n");
	    printf("Level must be <99 (Setting Level to 1)\n");
	    return (INPmkTemp
		    (" illegal (too high) argument to level parameter - level=1 assumed"));
	}

	return ((char *) NULL);
    }



    else {			/* no level on the line => default */
	*level = 1;
	printf("Warning -- Level not specified on line \"%s\"\nUsing level 1.\n", line);
	return ((char *) NULL);
    }




}
