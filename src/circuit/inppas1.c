/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include <stdio.h>

#include <config.h>
#include <ngspice.h>

#include "inppas1.h"

/*
 * The first pass of the circuit parser just looks for '.model' lines
 */

void
INPpas1(void *ckt, card *deck, INPtables *tab)
{
    card *current;
    char *INPdomodel(void *ckt, card *image, INPtables *tab);
    char *temp, *thisline;

    for(current = deck;current != NULL;current = current->nextcard) {
        /* SPICE-2 keys off of the first character of the line */
	thisline = current->line;

	while (*thisline && ((*thisline == ' ') || (*thisline == '\t')))
	    thisline++;

	if (*thisline == '.') {
            if(strncmp(thisline,".model",6)==0) { 
                temp = INPdomodel(ckt,current,tab);
                current->error = INPerrCat(current->error,temp);
            }
	}

	/* for now, we do nothing with the other cards - just 
	 * keep them in the list for pass 2 
	 */
    }
}
