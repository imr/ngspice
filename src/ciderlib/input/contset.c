/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/contdefs.h"
#include "ngspice/meshext.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

extern int CONTcheck( CONTcard * );
extern int CONTsetup( CONTcard *, ELCTelectrode * );


/*
 * Name:	CONTcheck
 * Purpose:	checks a list of CONTcards for input errors
 * Formals:	cardList: the list to check
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical device setup routines
 * Calls:	error message handler
 */
int
CONTcheck(CONTcard *cardList)
{
  CONTcard *card;
  int cardNum = 0;
  int error = OK;

  for ( card = cardList; card != NULL; card = card->CONTnextCard ) {
    cardNum++;
    if (!card->CONTnumberGiven) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "contact card %d is missing an electrode index", cardNum );
      error = E_PRIVATE;
    }

/* Return now if anything has failed */
    if (error) return(error);
  }
  return(OK);
}



/*
 * Name:	CONTsetup
 * Purpose:	copies information from list of CONTcard's to ELCTelectrode's
 * Formals:	cardList: list of cards to setup
 *		electrodeList: previously built list of ELCTelectrode's
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical devices
 * Calls:	CONTcheck
 */
int
CONTsetup(CONTcard *cardList, ELCTelectrode *electrodeList)
{
  CONTcard *card;
  ELCTelectrode *electrode;
  int error;

/* Check the card list */
  if ((error = CONTcheck( cardList )) != 0) return( error );

  for ( card = cardList; card != NULL; card = card->CONTnextCard ) {

    /* Copy workfunction to all matching electrodes */
    for ( electrode = electrodeList; electrode != NULL;
	electrode = electrode->next ) {
      if ( card->CONTnumber == electrode->id ) {
	if ( card->CONTworkfunGiven ) {
	  electrode->workf = card->CONTworkfun;
	} else {
	  electrode->workf = 4.10 /* electron volts */;
	}
      }
    }
  }
  return( OK );
}
