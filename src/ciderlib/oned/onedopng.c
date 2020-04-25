/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/profile.h"
#include "ngspice/macros.h"
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/cidersupt.h"

/* functions in this file are used to calculate the conc */

double 
ONEdopingValue(DOPprofile *pProfile, DOPtable *pTable, double x)
{
  double argX, argP, value=0.0;


  /* Find the appropriate lookup table if necessary */
  if (pProfile->type == LOOKUP) {
    while (pTable != NULL) {
      if (pTable->impId == pProfile->IMPID) {
	/* Found it */
	break;
      } else {
	pTable = pTable->next;
      }
    }
    if (pTable == NULL) {
      fprintf(stderr, "Error: unknown impurity profile %d\n",
	  ((int)pProfile->IMPID));
      controlled_exit(1);
    }
  }
  /* Find distances */
  if (pProfile->X_LOW > x) {
    argX = pProfile->X_LOW - x;
  } else if (x > pProfile->X_HIGH) {
    argX = x - pProfile->X_HIGH;
  } else {
    argX = 0.0;
  }

  argP = argX;

  /* Transform to coordinates of profile peak */
  argP -= pProfile->LOCATION;
  argP /= pProfile->CHAR_LENGTH;

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
      value = pProfile->CONC * (1.0 - argP);
    }
    break;
  case GAUSS:
    argP *= argP;
    if (argP > 80.0) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * exp(-argP);
    }
    break;
  case EXP:
    argP = ABS(argP);
    if (argP > 80.0) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * exp(-argP);
    }
    break;
  case ERRFC:
    argP = ABS(argP);
    if (argP > 10.0) {
      value = 0.0;
    } else {
      value = pProfile->PEAK_CONC * erfc(argP);
    }
    break;
  case LOOKUP:
    argP = ABS(argP);
    value = lookup(pTable->dopData, argP);
    break;
  default:
    break;
  }
  return (value);
}


void 
ONEsetDoping(ONEdevice *pDevice, DOPprofile *pProfile, DOPtable *pTable)
{
  ONEnode *pNode;
  ONEelem *pElem;
  DOPprofile *pP;
  double conc;
  int index, eIndex;
  BOOLEAN dopeMe;


  /* Clear doping info for all nodes. */
  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (index = 0; index <= 1; index++) {
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
  for (pP = pProfile; pP != NULL; pP = pP->next) {
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      if (pElem->elemType == SEMICON) {
	if (pP->numDomains > 0) {
	  dopeMe = FALSE;
	  for (index = 0; index < pP->numDomains; index++) {
	    if (pElem->domain == pP->domains[index]) {
	      dopeMe = TRUE;
	      break;
	    }
	  }
	} else {		/* domains not given, so dope all */
	  dopeMe = TRUE;
	}
	if (dopeMe) {
	  for (index = 0; index <= 1; index++) {
	    if (pElem->evalNodes[index]) {
	      pNode = pElem->pNodes[index];
	      conc = ONEdopingValue(pP, pTable, pNode->x);
	      pNode->netConc += conc;
	      if (conc < 0.0) {
		pNode->totalConc -= conc;
		pNode->na -= conc;
	      } else {
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
