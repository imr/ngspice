/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"

#include "inppas1.h"

/*
 * The first pass of the circuit parser just looks for '.model' lines
 */

void INPpas1(CKTcircuit *ckt, struct card *deck, INPtables * tab)
{
    struct card *current;
    char *temp, *thisline;

    for (current = deck; current != NULL; current = current->nextcard) {
	/* SPICE-2 keys off of the first character of the line */
	thisline = current->line;

	while (*thisline && ((*thisline == ' ') || (*thisline == '\t')))
	    thisline++;

	if (*thisline == '.') {
	    if (strncmp(thisline, ".model", 6) == 0) {
	      /* First check to see if model is multi-line.  If so,
		 read in all lines & stick them into tab. */
	      
#ifdef TRACE
	      /* SDB debug statement */
      	      printf("In INPpas1, handling line = %s \n", thisline); 
#endif

	      /* Now invoke INPdomodel to stick model into model table. */
		temp = INPdomodel(ckt, current, tab);
		current->error = INPerrCat(current->error, temp);
	    }
	}

	/* for now, we do nothing with the other cards - just 
	 * keep them in the list for pass 2 
	 */
    }
}
