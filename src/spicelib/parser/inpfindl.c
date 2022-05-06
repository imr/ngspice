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

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "inpxx.h"

char *INPfindLev(char *line, int *level)
{
    char *where;
    int error1;

    /*
     *where = line;
     */

    where = strstr(line, "level");

    if (where != NULL) {	/* found a level keyword on the line */

        where += 5;		/* skip the level keyword */
        while ((*where == ' ') || (*where == '\t') || (*where == '=') ||
           (*where == ',') || (*where == '(') || (*where == ')') ||
           (*where == '+'))
        {  /* legal white space - ignore */
            where++;
        }
        /* now the magic number,
           allow scientific notation of level, e.g. 4.900e1,
           offers only limited error checking.
         */
        *level = (int)(INPevaluate(&where, &error1, 0) + 0.5);

        if (*level < 0) {
            *level = 1;
            fprintf(stderr,"Illegal value for level.\n");
            fprintf(stderr,"Level must be >0 (Setting level to 1)\n");
            return (INPmkTemp
               (" illegal (negative) argument to level parameter - level=1 assumed"));
        }

        if (*level > 99) {	/* Limit to change in the future */
            *level = 1;
            fprintf(stderr,"Illegal value for level.\n");
            fprintf(stderr,"Level must be < 99 (Setting Level to 1)\n");
            return (INPmkTemp
               (" illegal (too high) argument to level parameter - level=1 assumed"));
        }

        return (NULL);
    }


    else {			/* no level on the line => default */
        *level = 1;
#ifdef TRACE			/* this is annoying for bjt's */
        fprintf(stderr,"Warning -- Level not specified on line \"%s\"\nUsing level 1.\n", line);
#endif
        return (NULL);
    }
}
