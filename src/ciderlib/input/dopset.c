/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modifed: 2001 Paolo Nenzi
**********/

#include "ngspice/carddefs.h"
#include "ngspice/cidersupt.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ciderinp.h"
#include "ngspice/cpextern.h"
#include "ngspice/dopdefs.h"
#include "ngspice/meshext.h"
#include "ngspice/ngspice.h"
#include "ngspice/numenum.h"
#include "ngspice/profile.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*
 * Name:	DOPcheck
 * Purpose:	checks a list of DOPcards for input errors
 * Formals:	cardList: the list to check
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical device setup routines
 * Calls:	error message handler
 */
int
DOPcheck(DOPcard *cardList, MESHcoord *xMeshList, MESHcoord *yMeshList)
{
  DOPcard *card;
  int cardNum = 0;
  int error = OK;

  for ( card = cardList; card != NULL; card = card->DOPnextCard ) {
    cardNum++;
    if (!card->DOPdomainsGiven) {
      card->DOPnumDomains = 0;
      card->DOPdomains = NULL;
    }
    if (!card->DOPprofileTypeGiven) {
      SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d does not specify profile type", cardNum );
      error = E_PRIVATE;
    } else switch (card->DOPprofileType) {
      case DOP_UNIF:
	if (!card->DOPconcGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs conc of uniform distribution", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_LINEAR:
	if (!card->DOPconcGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs peak conc of linear distribution", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_GAUSS:
	if (!card->DOPconcGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs peak conc of gaussian distribution", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_ERFC:
	if (!card->DOPconcGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs peak conc of error-function distribution", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_EXP:
	if (!card->DOPconcGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs peak conc of exponential distribution", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_SUPREM3:
      case DOP_SUPASCII:
	if (!card->DOPinFileGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs input-file name of suprem3 data", cardNum );
	  error = E_PRIVATE;
	}
	break;
      case DOP_ASCII:
	if (!card->DOPinFileGiven) {
	  SPfrontEnd->IFerrorf( ERR_WARNING, "doping card %d needs input-file name of ascii data", cardNum );
	  error = E_PRIVATE;
	}
	break;
      default:
	SPfrontEnd->IFerrorf( ERR_FATAL, "doping card %d has unrecognized profile type", cardNum );
	error = E_NOTFOUND;
	break;
    }
    if (!card->DOProtateLatGiven) {
      card->DOProtateLat = FALSE;
    }
    if (!card->DOPlatProfileTypeGiven || card->DOProtateLat) {
      card->DOPlatProfileType = card->DOPprofileType;
    }
    if (!card->DOPratioLatGiven) {
      card->DOPratioLat = 1.0;
    }
    if (!card->DOPcharLenGiven) {
      card->DOPcharLen = 1.0e-4; 	/* 1um in centimeters */
    }
    if (!card->DOPlocationGiven) {
      card->DOPlocation = 0.0;
    }
    if (!card->DOPimpurityTypeGiven) {
      card->DOPimpurityType = IMP_N_TYPE;
    } else switch (card->DOPimpurityType) {
      case DOP_BORON:
	card->DOPimpurityType = IMP_BORON;
	break;
      case DOP_PHOSP:
	card->DOPimpurityType = IMP_PHOSPHORUS;
	break;
      case DOP_ARSEN:
	card->DOPimpurityType = IMP_ARSENIC;
	break;
      case DOP_ANTIM:
	card->DOPimpurityType = IMP_ANTIMONY;
	break;
      case DOP_N_TYPE:
	card->DOPimpurityType = IMP_N_TYPE;
	break;
      case DOP_P_TYPE:
	card->DOPimpurityType = IMP_P_TYPE;
	break;
      default:
	break;
    }

    if (!card->DOPaxisTypeGiven) {
      if ( xMeshList && yMeshList ) { /* both lists are non-empty */
	card->DOPaxisType = DOP_Y_AXIS;
      } else if ( xMeshList ) { /* x-mesh list is non-empty */
	card->DOPaxisType = DOP_X_AXIS;
      } else if ( yMeshList ) { /* y-mesh list is non-empty */
	card->DOPaxisType = DOP_Y_AXIS;
      }
    }

/* Return now if anything has failed */
    if (error) return(error);
  }
  return(OK);
}



/*
 * Name:	DOPsetup
 * Purpose:	convert a list of DOPcard's to DOPprofile's
 * Formals:	cardList: list of cards to setup
 *		profileList: returns the list of DOPprofile's
 *		xMeshList: list of coordinates in the x mesh
 *		yMeshList: list of coordinates in the y mesh
 * Returns:	OK/E_PRIVATE
 * Users:	 numerical devices
 * Calls:	DOPcheck
 */
int
DOPsetup(DOPcard *cardList, DOPprofile **profileList, DOPtable **tableList,
         MESHcoord *xMeshList, MESHcoord *yMeshList)
{
  DOPcard *card;
  DOPprofile *newProfile = NULL, *endProfile;
  int impurityId = 0;
  double xMin, xMax, yMin, yMax;
  double sign;
  int error, xProfUnif, yProfUnif;

/* Initialize list of profiles */
  *profileList = endProfile = NULL;

/* Check the card list */
  if ((error = DOPcheck( cardList, xMeshList, yMeshList )) != 0) return( error );

/* Find the limits on locations */
  MESHlBounds( xMeshList, &xMin, &xMax );
  MESHlBounds( yMeshList, &yMin, &yMax );

  for ( card = cardList; card != NULL; card = card->DOPnextCard ) {

    if (*profileList == NULL) {
      RALLOC( newProfile, DOPprofile, 1 );
      *profileList = newProfile;
    }
    else {
      RALLOC( newProfile->next, DOPprofile, 1 );
      newProfile = newProfile->next;
    }
    newProfile->next = NULL;

    newProfile->numDomains = card->DOPnumDomains;
    if ( newProfile->numDomains > 0 ) {
      int i;
      RALLOC( newProfile->domains, int, newProfile->numDomains );
      for ( i=0; i < newProfile->numDomains; i++ ) {
	newProfile->domains[i] = card->DOPdomains[i];
      }
    }
    else {
      newProfile->domains = NULL;
    }

    if ( card->DOPimpurityType == IMP_P_TYPE ) {
      sign = -1.0;
    }
    else {
      sign = 1.0;
    }
    switch( card->DOPprofileType ) {
      case DOP_UNIF:
	newProfile->type = UNIF;
	newProfile->CONC = sign * card->DOPconc;
	break;
      case DOP_LINEAR:
	newProfile->type = LIN;
	newProfile->CONC = sign * card->DOPconc;
	break;
      case DOP_GAUSS:
	newProfile->type = GAUSS;
	newProfile->CONC = sign * card->DOPconc;
	break;
      case DOP_ERFC:
	newProfile->type = ERRFC;
	newProfile->CONC = sign * card->DOPconc;
	break;
      case DOP_EXP:
	newProfile->type = EXP;
	newProfile->CONC = sign * card->DOPconc;
	break;
      case DOP_SUPREM3:
	newProfile->type = LOOKUP;
	if (readSupremData( card->DOPinFile, 0, card->DOPimpurityType,
            tableList) != 0) {
        (void) fprintf(cp_err, "Doping setup failed.\n");
        return -1;
    }
	newProfile->IMPID = ++impurityId;
	break;
      case DOP_SUPASCII:
	newProfile->type = LOOKUP;
    if (readSupremData( card->DOPinFile, 1, card->DOPimpurityType,
            tableList) != 0) {
        (void) fprintf(cp_err, "Doping setup failed.\n");
        return -1;
    }
	newProfile->IMPID = ++impurityId;
	break;
      case DOP_ASCII:
	newProfile->type = LOOKUP;
    if (readAsciiData(card->DOPinFile, card->DOPimpurityType,
            tableList) != 0) {
        (void) fprintf(cp_err, "Doping setup failed.\n");
        return -1;
    }
	newProfile->IMPID = ++impurityId;
	break;
      default:
	break;
    }
    switch( card->DOPlatProfileType ) {
      case DOP_UNIF:
	newProfile->latType = UNIF;
	break;
      case DOP_LINEAR:
	newProfile->latType = LIN;
	break;
      case DOP_GAUSS:
	newProfile->latType = GAUSS;
	break;
      case DOP_ERFC:
	newProfile->latType = ERRFC;
	break;
      case DOP_EXP:
	newProfile->latType = EXP;
	break;
      case DOP_SUPREM3:
      case DOP_SUPASCII:
	newProfile->latType = LOOKUP;
	break;
      case DOP_ASCII:
	newProfile->latType = LOOKUP;
	break;
      default:
	break;
    }
    newProfile->rotate = card->DOProtateLat;
    newProfile->LOCATION = card->DOPlocation;
    newProfile->CHAR_LENGTH = card->DOPcharLen;
    newProfile->LAT_RATIO = card->DOPratioLat;

    xProfUnif = yProfUnif = FALSE;
    if (card->DOPaxisType == DOP_X_AXIS) {
      newProfile->DIRECTION = X;
      if (newProfile->type == UNIF) xProfUnif = TRUE;
      if (newProfile->latType == UNIF) yProfUnif = TRUE;
    } 
    else {
      newProfile->DIRECTION = Y;
      if (newProfile->type == UNIF) yProfUnif = TRUE;
      if (newProfile->latType == UNIF) xProfUnif = TRUE;
    }

/* Fill in x coordinates.  Use defaults if necessary */
    if (card->DOPxLowGiven && card->DOPxHighGiven) {
      newProfile->X_LOW = card->DOPxLow;
      newProfile->X_HIGH = card->DOPxHigh;
    }
    else if (card->DOPxLowGiven) {
      newProfile->X_LOW = card->DOPxLow;
      if (xProfUnif) {
	newProfile->X_HIGH = xMax;
      }
      else {
	newProfile->X_HIGH = newProfile->X_LOW;
      }
    }
    else if (card->DOPxHighGiven) {
      newProfile->X_HIGH = card->DOPxHigh;
      if (xProfUnif) {
	newProfile->X_LOW = xMin;
      }
      else {
	newProfile->X_LOW = newProfile->X_HIGH;
      }
    }
    else {
      if (xProfUnif) {
	newProfile->X_LOW = xMin;
	newProfile->X_HIGH = xMax;
      }
      else {
	newProfile->X_LOW = 0.5 * (xMin + xMax);
	newProfile->X_HIGH = 0.5 * (xMin + xMax);
      }
    }

/* Fill in y coordinates.  Use defaults if necessary */
    if (card->DOPyLowGiven && card->DOPyHighGiven) {
      newProfile->Y_LOW = card->DOPyLow;
      newProfile->Y_HIGH = card->DOPyHigh;
    }
    else if (card->DOPyLowGiven) {
      newProfile->Y_LOW = card->DOPyLow;
      if (yProfUnif) {
	newProfile->Y_HIGH = yMax;
      }
      else {
	newProfile->Y_HIGH = newProfile->Y_LOW;
      }
    }
    else if (card->DOPyHighGiven) {
      newProfile->Y_HIGH = card->DOPyHigh;
      if (xProfUnif) {
	newProfile->Y_LOW = yMin;
      }
      else {
	newProfile->Y_LOW = newProfile->Y_HIGH;
      }
    }
    else {
      if (yProfUnif) {
	newProfile->Y_LOW = yMin;
	newProfile->Y_HIGH = yMax;
      }
      else {
	newProfile->Y_LOW = 0.5 * (yMin + yMax);
	newProfile->Y_HIGH = 0.5 * (yMin + yMax);
      }
    }
  }
  return( OK );
}
