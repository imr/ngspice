/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/bdrydefs.h"
#include "ngspice/meshext.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"

extern int BDRYcheck( BDRYcard *, DOMNdomain * );
extern int BDRYsetup( BDRYcard *, MESHcoord *, MESHcoord *, DOMNdomain * );


/*
 * Name:	BDRYcheck
 * Purpose:	checks a list of BDRYcards for input errors
 * Formals:	cardList: the list to check
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical device setup routines
 * Calls:	error message handler
 */
int
BDRYcheck(BDRYcard *cardList, DOMNdomain *domnList)
{
  BDRYcard *card;
  DOMNdomain *domn;
  int cardNum = 0;
  int error = OK;

  for ( card = cardList; card != NULL; card = card->BDRYnextCard ) {
    cardNum++;
    if (card->BDRYxLowGiven && card->BDRYixLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "boundary card %d uses both location and index - location ignored", cardNum );
      card->BDRYxLowGiven = FALSE;
    }
    if (card->BDRYxHighGiven && card->BDRYixHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "boundary card %d uses both location and index - location ignored", cardNum );
      card->BDRYxHighGiven = FALSE;
    }
    if (card->BDRYyLowGiven && card->BDRYiyLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "boundary card %d uses both location and index - location ignored", cardNum );
      card->BDRYyLowGiven = FALSE;
    }
    if (card->BDRYyHighGiven && card->BDRYiyHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "boundary card %d uses both location and index - location ignored", cardNum );
      card->BDRYyHighGiven = FALSE;
    }
    if (!card->BDRYdomainGiven) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "boundary card %d is missing a domain index", cardNum );
      error = E_PRIVATE;
    } else {
      /* Make sure the domain exists */
      for ( domn = domnList; domn != NULL; domn = domn->next ) {
	if ( card->BDRYdomain == domn->id ) {
	  break;
	}
      }
      if (domn == NULL) {
	SPfrontEnd->IFerrorf( ERR_WARNING, "boundary card %d specifies a non-existent domain", cardNum );
	error = E_PRIVATE;
      }
    }

    if (!card->BDRYneighborGiven) {
      card->BDRYneighbor = card->BDRYdomain;
    } else {
      /* Make sure the neighbor exists */
      for ( domn = domnList; domn != NULL; domn = domn->next ) {
	if ( card->BDRYneighbor == domn->id ) {
	  break;
	}
      }
      if (domn == NULL) {
	SPfrontEnd->IFerrorf( ERR_WARNING, "interface card %d specifies a non-existent domain", cardNum );
	error = E_PRIVATE;
      }
    }

    if (!card->BDRYqfGiven) {
      card->BDRYqf = 0.0;
    }
    if (!card->BDRYsnGiven) {
      card->BDRYsn = 0.0;
    }
    if (!card->BDRYspGiven) {
      card->BDRYsp = 0.0;
    }
    if (!card->BDRYlayerGiven) {
      card->BDRYlayer = 0.0;
    }

/* Return now if anything has failed */
    if (error) return(error);
  }
  return(OK);
}



/*
 * Name:	BDRYsetup
 * Purpose:	Checks BDRY cards and then sets the indices
 * Formals:	cardList: list of cards to setup, returns with indices set
 *		xMeshList: list of coordinates in the x mesh
 *		yMeshList: list of coordinates in the y mesh
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical devices
 * Calls:	BDRYcheck
 */
int
BDRYsetup(BDRYcard *cardList, MESHcoord *xMeshList, MESHcoord *yMeshList,DOMNdomain *domnList)
{
  BDRYcard *card;
  int ixMin, ixMax, iyMin, iyMax;
  int cardNum = 0;
  int error;

/* Check the card list */
  if ((error = BDRYcheck( cardList, domnList )) != 0) return( error );

/* Find the limits on the indices */
  MESHiBounds( xMeshList, &ixMin, &ixMax );
  MESHiBounds( yMeshList, &iyMin, &iyMax );

  error = OK;
  for ( card = cardList; card != NULL; card = card->BDRYnextCard ) {
    cardNum++;

    if (card->BDRYixLowGiven) {
      card->BDRYixLow = MAX(card->BDRYixLow, ixMin);
    }
    else if (card->BDRYxLowGiven) {
      card->BDRYixLow = MESHlocate( xMeshList, card->BDRYxLow );
    }
    else {
      card->BDRYixLow = ixMin;
    }
    if (card->BDRYixHighGiven) {
      card->BDRYixHigh = MIN(card->BDRYixHigh, ixMax);
    }
    else if (card->BDRYxHighGiven) {
      card->BDRYixHigh = MESHlocate( xMeshList, card->BDRYxHigh );
    }
    else {
      card->BDRYixHigh = ixMax;
    }
    if (card->BDRYixLow > card->BDRYixHigh) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "boundary card %d has low x index (%d) > high x index (%d)", cardNum, card->BDRYixHigh, card->BDRYixLow );
      error = E_PRIVATE;
    }
    if (card->BDRYiyLowGiven) {
      card->BDRYiyLow = MAX(card->BDRYiyLow, iyMin);
    }
    else if (card->BDRYyLowGiven) {
      card->BDRYiyLow = MESHlocate( yMeshList, card->BDRYyLow );
    }
    else {
      card->BDRYiyLow = iyMin;
    }
    if (card->BDRYiyHighGiven) {
      card->BDRYiyHigh = MIN(card->BDRYiyHigh, iyMax);
    }
    else if (card->BDRYyHighGiven) {
      card->BDRYiyHigh = MESHlocate( yMeshList, card->BDRYyHigh );
    }
    else {
      card->BDRYiyHigh = iyMax;
    }
    if (card->BDRYiyLow > card->BDRYiyHigh) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "boundary card %d has low y index (%d) > high y index (%d)", cardNum, card->BDRYiyHigh, card->BDRYiyLow );
      error = E_PRIVATE;
    }
  }
  return( error );
}
