/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/profile.h"
#include "ngspice/macros.h"
#include "ngspice/bool.h"
#include "twoddefs.h"
#include "twodext.h"
#include "ngspice/cidersupt.h"

/* functions in this file are used to calculate the conc */

double 
TWOdopingValue(DOPprofile *pProfile, DOPtable *pTable, double x, 
               double y)
{
  double argX, argY, argP, argL, value = 0.0;
  
  /* Find the appropriate lookup table if necessary */
  if (pProfile->type == LOOKUP) {
    while ( pTable != NULL ) {
      if (pTable->impId == pProfile->IMPID) {
        /* Found it */
	break;
      } else {
	pTable = pTable->next;
      }
    }
    if ( pTable == NULL ) {
      fprintf( stderr, "Error: unknown impurity profile %d\n",
	      ((int)pProfile->IMPID) );
      controlled_exit(1);
    }
  }
  /* Find distances */
  if ( pProfile->Y_LOW > y ) {
    argY = pProfile->Y_LOW - y;
  } else if ( y > pProfile->Y_HIGH ) {
    argY = y - pProfile->Y_HIGH;
  } else {
    argY = 0.0;
  }
  if ( pProfile->X_LOW > x ) {
    argX = pProfile->X_LOW - x;
  } else if ( x > pProfile->X_HIGH ) {
    argX = x - pProfile->X_HIGH;
  } else {
    argX = 0.0;
  }
  
  if ( pProfile->DIRECTION == Y ) {
    argP = argY;
    argL = argX / pProfile->LAT_RATIO;
  }
  else {
    argP = argX;
    argL = argY / pProfile->LAT_RATIO;
  }
  if ( pProfile->rotate ) {
    argP = hypot(argP, argL);
    argL = 0.0;
  }
  
  /* Transform to coordinates of profile peak */
  argP -= pProfile->LOCATION;
  argP /= pProfile->CHAR_LENGTH;
  argL -= pProfile->LOCATION;
  argL /= pProfile->CHAR_LENGTH;
  
  switch (pProfile->type) {
  case UNIF:
    if (argP > 0.0) {
      value = 0.0;
    } else {
      value = pProfile->CONC;
    }
    break;
  case LIN:
    argP = ABS(argP);
    if (argP > 1.0) {
      value = 0.0;
    } else {
      value = pProfile->CONC * ( 1.0 - argP );
    }
    break;
  case GAUSS:
    argP *= argP;
    if ( argP > 80.0 ) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * exp( -argP );
    }
    break;
  case EXP:
    argP = ABS(argP);
    if ( argP > 80.0 ) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * exp( -argP );
    }
    break;
  case ERRFC:
    argP = ABS(argP);
    if ( argP > 10.0 ) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * erfc( argP );
    }
    break;
  case LOOKUP:
    argP = ABS(argP);
    value = lookup( pTable->dopData, argP );
    break;
  default:
    break;
  }
  if (!pProfile->rotate) { /* Tensor product in lateral direction */
    switch (pProfile->latType) {
    case UNIF:
      if (argL > 0.0) {
	value = 0.0;
      }
      break;
    case LIN:
      argL = ABS(argL);
      if (argL > 1.0) {
	value = 0.0;
      } else {
	value *= ( 1.0 - argL );
      }
      break;
    case GAUSS:
      argL *= argL;
      if ( argL > 80.0 ) {
	value = 0.0;
      } else {
	value *= exp( -argL );
      }
      break;
    case EXP:
      argL = ABS(argL);
      if ( argL > 80.0 ) {
	value = 0.0;
      } else {
	value *= exp( -argL );
      }
      break;
    case ERRFC:
      argL = ABS(argL);
      if ( argP > 10.0 ) {
	value = 0.0;
      } else {
	value *= erfc( argL );
      }
      break;
    case LOOKUP:
      argL = ABS(argL);
      value *= lookup( pTable->dopData, argL ) / lookup( pTable->dopData, 0.0 );
      break;
    default:
      break;
    }
  } /* end: not rotated */
  return( value );
}

void 
TWOsetDoping(TWOdevice *pDevice, DOPprofile *pProfile, DOPtable *pTable)
{
  TWOnode *pNode;
  TWOelem *pElem;
  DOPprofile *pP;
  double conc;
  int index, eIndex;
  BOOLEAN dopeMe;

  /* Clear doping info for all nodes. */
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for (index = 0; index <= 3; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	pNode->na = 0.0;
	pNode->nd = 0.0;
	pNode->netConc = 0.0;
	pNode->totalConc = 0.0;
      }
    }
  }
  /* Now compute the contribution to the total doping from each profile. */
  for ( pP = pProfile; pP != NULL; pP = pP->next ) {
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      if ( pElem->elemType == SEMICON ) {
	if ( pP->numDomains > 0 ) {
	  dopeMe = FALSE;
	  for ( index = 0; index < pP->numDomains; index++ ) {
	    if ( pElem->domain == pP->domains[ index ] ) {
	      dopeMe = TRUE;
	      break;
	    }
	  }
	} else { /* domains not given, so dope all */
	  dopeMe = TRUE;
	}
	if ( dopeMe ) {
	  for ( index = 0; index <= 3; index++ ) {
	    if ( pElem->evalNodes[ index ] ) {
	      pNode = pElem->pNodes[ index ];
	      conc = TWOdopingValue( pP, pTable,
		  pDevice->xScale[ pNode->nodeI ],
		  pDevice->yScale[ pNode->nodeJ ] );
	      pNode->netConc += conc;
	      if ( conc < 0.0 ) {
		pNode->totalConc -= conc;
		pNode->na -= conc;
	      }
	      else {
		pNode->totalConc += conc;
		pNode->nd += conc;
	      }
	    }
	  }
	}
      }
    }
  }
}
