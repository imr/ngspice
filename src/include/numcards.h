/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1991 David Gates
**********/
/*
 * Structures for parsing numerical-device input cards
 */
 
 /* Member of CIDER device simulator
 * Version: 1b1
 */
 
#ifndef NUMCARDS_H
#define NUMCARDS_H

#include "ifsim.h"

/*
 * Generic header for a linked list of cards 
 */

typedef struct sGENcard {
    struct sGENcard *GENnextCard; /* pointer to next card of this type */
} GENcard;

/*
 * Structure: IFcardInfo
 *  
 * This structure is a generic description of an input card to
 * the program.  It can be used in situations where input parameters
 * need to be grouped together under a single heading.
 */


typedef struct sIFcardInfo {
    char *name;			/* name of the card */
    char *description;		/* description of its purpose */

    int numParms;		/* number of parameter descriptors */
    IFparm *cardParms;		/* array  of parameter descriptors */

    int (*newCard)(void**,void*);
	/* routine to add a new card to a numerical device model */
    int (*setCardParm)(int,IFvalue*,void*);
	/* routine to input a parameter to a card instance */
    int (*askCardQuest)(int,IFvalue*,void*);
	/* routine to find out about a card's details */
} IFcardInfo;

#endif /* NUMCARDS_H */
