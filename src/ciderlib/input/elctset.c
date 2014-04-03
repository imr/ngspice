/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/elctdefs.h"
#include "ngspice/meshext.h"
#include "ngspice/twomesh.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


extern int ELCTcheck( ELCTcard * );
extern int ELCTsetup( ELCTcard *, ELCTelectrode **, MESHcoord *, MESHcoord * );


/*
 * Name:	ELCTcheck
 * Purpose:	checks a list of ELCTcards for input errors
 * Formals:	cardList: the list to check
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical device setup routines
 * Calls:	error message handler
 */
int
ELCTcheck(ELCTcard *cardList)
{
  ELCTcard *card;
  int cardNum = 0;
  int error = OK;

  for ( card = cardList; card != NULL; card = card->ELCTnextCard ) {
    cardNum++;
    if (card->ELCTxLowGiven && card->ELCTixLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "electrode card %d uses both location and index - location ignored", cardNum );
      card->ELCTxLowGiven = FALSE;
    }
    if (card->ELCTxHighGiven && card->ELCTixHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "electrode card %d uses both location and index - location ignored", cardNum );
      card->ELCTxHighGiven = FALSE;
    }
    if (card->ELCTyLowGiven && card->ELCTiyLowGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "electrode card %d uses both location and index - location ignored", cardNum );
      card->ELCTyLowGiven = FALSE;
    }
    if (card->ELCTyHighGiven && card->ELCTiyHighGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "electrode card %d uses both location and index - location ignored", cardNum );
      card->ELCTyHighGiven = FALSE;
    }
    if (!card->ELCTnumberGiven) {
      card->ELCTnumber = -1;
    }

/* Return now if anything has failed */
    if (error) return(error);
  }
  return(OK);
}


/*
 * Name:	ELCTsetup
 * Purpose:	convert a list of ELCTcard's to ELCTelectrode's
 * Formals:	cardList: list of cards to setup
 *		electrodeList: returns the list of ELCTelectrode's
 *		xMeshList: list of coordinates in the x mesh
 *		yMeshList: list of coordinates in the y mesh
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical devices
 * Calls:	ELCTcheck
 */
int
ELCTsetup(ELCTcard *cardList, ELCTelectrode **electrodeList, 
          MESHcoord *xMeshList, MESHcoord *yMeshList)
{
  ELCTcard *card;
  ELCTelectrode *newElectrode = NULL;
  int ixMin, ixMax, iyMin, iyMax;
  int cardNum = 0;
  int error;

/* Initialize list of electrodes */
  *electrodeList = NULL;

/* Check the card list */
  if ((error = ELCTcheck( cardList )) != 0) return( error );

/* Find the limits on the indices */
  MESHiBounds( xMeshList, &ixMin, &ixMax );
  MESHiBounds( yMeshList, &iyMin, &iyMax );

  error = OK;
  for ( card = cardList; card != NULL; card = card->ELCTnextCard ) {
    cardNum++;

    if (*electrodeList == NULL) {
      RALLOC( newElectrode, ELCTelectrode, 1 );
      *electrodeList = newElectrode;
    } else {
      RALLOC( newElectrode->next, ELCTelectrode, 1 );
      newElectrode = newElectrode->next;
    }
    newElectrode->next = NULL;
    newElectrode->id = card->ELCTnumber;
    newElectrode->workf = 4.10 /* electron volts */;

    if (card->ELCTixLowGiven) {
      newElectrode->ixLo = MAX(card->ELCTixLow, ixMin);
    }
    else if (card->ELCTxLowGiven) {
      newElectrode->ixLo = MESHlocate( xMeshList, card->ELCTxLow );
    }
    else {
      newElectrode->ixLo = ixMin;
    }
    if (card->ELCTixHighGiven) {
      newElectrode->ixHi = MIN(card->ELCTixHigh, ixMax);
    }
    else if (card->ELCTxHighGiven) {
      newElectrode->ixHi = MESHlocate( xMeshList, card->ELCTxHigh );
    }
    else {
      newElectrode->ixHi = ixMax;
    }
    if (newElectrode->ixLo > newElectrode->ixHi) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "electrode card %d has low x index (%d) > high x index (%d)", cardNum, newElectrode->ixLo, newElectrode->ixHi );
      error = E_PRIVATE;
    }
    if (card->ELCTiyLowGiven) {
      newElectrode->iyLo = MAX(card->ELCTiyLow, iyMin);
    }
    else if (card->ELCTyLowGiven) {
      newElectrode->iyLo = MESHlocate( yMeshList, card->ELCTyLow );
    }
    else {
      newElectrode->iyLo = iyMin;
    }
    if (card->ELCTiyHighGiven) {
      newElectrode->iyHi = MIN(card->ELCTiyHigh, iyMax);
    }
    else if (card->ELCTyHighGiven) {
      newElectrode->iyHi = MESHlocate( yMeshList, card->ELCTyHigh );
    }
    else {
      newElectrode->iyHi = iyMax;
    }
    if (newElectrode->iyLo > newElectrode->iyHi) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "electrode card %d has low y index (%d) > high y index (%d)", cardNum, newElectrode->iyLo, newElectrode->iyHi );
      error = E_PRIVATE;
    }
  }
  return( error );
}
