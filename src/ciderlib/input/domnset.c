/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modifed: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/numenum.h"
#include "ngspice/domndefs.h"
#include "ngspice/material.h"
#include "ngspice/meshext.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


extern int DOMNcheck( DOMNcard *, MATLmaterial * );
extern int DOMNsetup( DOMNcard *, DOMNdomain **, MESHcoord *, MESHcoord *,
		      MATLmaterial * );


/*
 * Name:	DOMNcheck
 * Purpose:	checks a list of DOMNcards for input errors
 * Formals:	cardList: the list to check
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical device setup routines
 * Calls:	error message handler
 */
int
DOMNcheck(DOMNcard *cardList, MaterialInfo *matlList)
{
  DOMNcard *card;
  MATLmaterial *matl;
  int cardNum = 0;
  int error = OK;

  for ( card = cardList; card != NULL; card = card->DOMNnextCard ) {
    cardNum++;
    if (card->DOMNxLowGiven && card->DOMNixLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "domain card %d uses both location and index - location ignored", cardNum );
      card->DOMNxLowGiven = FALSE;
    }
    if (card->DOMNxHighGiven && card->DOMNixHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "domain card %d uses both location and index - location ignored", cardNum );
      card->DOMNxHighGiven = FALSE;
    }
    if (card->DOMNyLowGiven && card->DOMNiyLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "domain card %d uses both location and index - location ignored", cardNum );
      card->DOMNyLowGiven = FALSE;
    }
    if (card->DOMNyHighGiven && card->DOMNiyHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "domain card %d uses both location and index - location ignored", cardNum );
      card->DOMNyHighGiven = FALSE;
    }
    if (!card->DOMNmaterialGiven) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "domain card %d is missing a material index", cardNum );
      error = E_PRIVATE;
    } else {
      /* Make sure the material exists */
      for ( matl = matlList; matl != NULL; matl = matl->next ) {
	if ( card->DOMNmaterial == matl->id ) {
	  break;
	}
      }
      if (matl == NULL) {
	SPfrontEnd->IFerrorf( ERR_WARNING, "domain card %d specifies a non-existent material", cardNum );
	error = E_PRIVATE;
      }
    }
    if (!card->DOMNnumberGiven) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "domain card %d is missing an ID number", cardNum );
      error = E_PRIVATE;
    }

/* Return now if anything has failed */
    if (error) return(error);
  }
  return(OK);
}



/*
 * Name:	DOMNsetup
 * Purpose:	convert a list of DOMNcard's to DOMNdomain's
 * Formals:	cardList: list of cards to setup
 *		domainList: returns the list of DOMNdomain's
 *		xMeshList: list of coordinates in the x mesh
 *		yMeshList: list of coordinates in the y mesh
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical devices
 * Calls:	DOMNcheck
 */
int
DOMNsetup(DOMNcard *cardList, DOMNdomain **domainList, MESHcoord *xMeshList, 
          MESHcoord *yMeshList, MaterialInfo *materialList)
{
  DOMNcard *card;
  DOMNdomain *newDomain = NULL;
  int ixMin, ixMax, iyMin, iyMax;
  int cardNum = 0;
  int error;

/* Initialize list of domains */
  *domainList = NULL;

/* Check the card list */
  if ((error = DOMNcheck( cardList, materialList )) != 0) return( error );

/* Find the limits on the indices */
  MESHiBounds( xMeshList, &ixMin, &ixMax );
  MESHiBounds( yMeshList, &iyMin, &iyMax );

  error = OK;
  for ( card = cardList; card != NULL; card = card->DOMNnextCard ) {
    cardNum++;

    if (*domainList == NULL) {
      RALLOC( newDomain, DOMNdomain, 1 );
      *domainList = newDomain;
    } else {
      RALLOC( newDomain->next, DOMNdomain, 1 );
      newDomain = newDomain->next;
    }

    newDomain->id = card->DOMNnumber;
    newDomain->material = card->DOMNmaterial;
    newDomain->next = NULL;

    if (card->DOMNixLowGiven) {
      newDomain->ixLo = MAX(card->DOMNixLow, ixMin);
    }
    else if (card->DOMNxLowGiven) {
      newDomain->ixLo = MESHlocate( xMeshList, card->DOMNxLow );
    }
    else {
      newDomain->ixLo = ixMin;
    }
    if (card->DOMNixHighGiven) {
      newDomain->ixHi = MIN(card->DOMNixHigh, ixMax);
    }
    else if (card->DOMNxHighGiven) {
      newDomain->ixHi = MESHlocate( xMeshList, card->DOMNxHigh );
    }
    else {
      newDomain->ixHi = ixMax;
    }
    if (newDomain->ixLo > newDomain->ixHi) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "domain card %d has low x index (%d) > high x index (%d)", cardNum, newDomain->ixLo, newDomain->ixHi );
      error = E_PRIVATE;
    }
    if (card->DOMNiyLowGiven) {
      newDomain->iyLo = MAX(card->DOMNiyLow, iyMin);
    }
    else if (card->DOMNyLowGiven) {
      newDomain->iyLo = MESHlocate( yMeshList, card->DOMNyLow );
    }
    else {
      newDomain->iyLo = iyMin;
    }
    if (card->DOMNiyHighGiven) {
      newDomain->iyHi = MIN(card->DOMNiyHigh, iyMax);
    }
    else if (card->DOMNyHighGiven) {
      newDomain->iyHi = MESHlocate( yMeshList, card->DOMNyHigh );
    }
    else {
      newDomain->iyHi = iyMax;
    }
    if (newDomain->iyLo > newDomain->iyHi) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "domain card %d has low y index (%d) > high y index (%d)", cardNum, newDomain->iyLo, newDomain->iyHi );
      error = E_PRIVATE;
    }
  }
  return( error );
}
