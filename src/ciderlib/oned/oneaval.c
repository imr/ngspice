/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/macros.h"

#include "onedext.h"
#include "oneddefs.h"

double 
ONEavalanche(BOOLEAN rhsOnly, ONEdevice *pDevice, ONEnode *pNode)
{
  ONEelem *pLElem, *pRElem;
  ONEedge *pLEdge, *pREdge;
  int numNodes = pDevice->numNodes;
  double dJnDpsiPrev, dJpDpsiPrev;
  double eField, temp, jn, jp;
  double signE, signN, signP, coeffR, coeffL, alphaN, alphaP;
  double generation = 0.0;
  double dAlphaNDpsiM1, dAlphaNDpsi, dAlphaNDpsiP1;
  double dAlphaPDpsiM1, dAlphaPDpsi, dAlphaPDpsiP1;
  ONEmaterial *info;


  pRElem = pNode->pRightElem;
  pLElem = pNode->pLeftElem;

  if (pRElem->evalNodes[0]) {
    info = pRElem->matlInfo;
  } else {
    info = pLElem->matlInfo;
  }

  pREdge = pRElem->pEdge;
  pLEdge = pLElem->pEdge;
  dJnDpsiPrev = pLEdge->dJnDpsiP1;
  dJpDpsiPrev = pLEdge->dJpDpsiP1;

  temp = pRElem->dx + pLElem->dx;
  coeffR = pLElem->dx / temp;
  coeffL = pRElem->dx / temp;

  eField = -(coeffR * pREdge->dPsi * pRElem->rDx +
      coeffL * pLEdge->dPsi * pLElem->rDx);
  jn = coeffR * pREdge->jn + coeffL * pLEdge->jn;
  jp = coeffR * pREdge->jp + coeffL * pLEdge->jp;

  signE = SGN(eField);
  eField = ABS(eField);
  if (eField == 0.0) {
    return (0.0);
  }
  signN = SGN(jn);
  if (signN * signE > 0.0) {
    /* field accelerates the carriers, hence avalanche */
    if (info->bii[ELEC] / eField > 80.0) {
      alphaN = 0.0;
    } else {
      alphaN = info->aii[ELEC] * exp(-info->bii[ELEC] / eField);
    }
  } else {
    alphaN = 0.0;
  }
  signP = SGN(jp);
  if (signP * signE > 0.0) {
    /* field accelerates the carriers, hence avalanche */
    if (info->bii[HOLE] / eField > 80.0) {
      alphaP = 0.0;
    } else {
      alphaP = info->aii[HOLE] * exp(-info->bii[HOLE] / eField);
    }
  } else {
    alphaP = 0.0;
  }

  if ((alphaN == 0.0) && (alphaP == 0.0)) {
    return (generation);
  }
  generation = (alphaN * ABS(jn) + alphaP * ABS(jp)) *
      0.5 * (pRElem->dx + pLElem->dx);
  if (rhsOnly) {
    return (generation);
  }
  if (alphaN == 0.0) {
    dAlphaNDpsiM1 = 0.0;
    dAlphaNDpsiP1 = 0.0;
    dAlphaNDpsi = 0.0;
  } else {
    temp = alphaN * info->bii[ELEC] / (eField * eField);
    dAlphaNDpsiM1 = signE * temp * (coeffL * pLElem->rDx);
    dAlphaNDpsiP1 = -signE * temp * (coeffR * pRElem->rDx);
    dAlphaNDpsi = -(dAlphaNDpsiM1 + dAlphaNDpsiP1);
  }

  if (alphaP == 0.0) {
    dAlphaPDpsiM1 = 0.0;
    dAlphaPDpsiP1 = 0.0;
    dAlphaPDpsi = 0.0;
  } else {
    temp = alphaP * info->bii[HOLE] / (eField * eField);
    dAlphaPDpsiM1 = signE * temp * (coeffL * pLElem->rDx);
    dAlphaPDpsiP1 = -signE * temp * (coeffR * pRElem->rDx);
    dAlphaPDpsi = -(dAlphaPDpsiM1 + dAlphaPDpsiP1);
  }

  coeffR = 0.5 * pLElem->dx;
  coeffL = 0.5 * pRElem->dx;

  if (pNode->nodeI != 2) {
    *(pNode->fNPsiiM1) +=
	signN * (-alphaN * coeffL * dJnDpsiPrev +
	coeffL * pLEdge->jn * dAlphaNDpsiM1) +
	signP * (-alphaP * coeffL * dJpDpsiPrev +
	coeffL * pLEdge->jp * dAlphaPDpsiM1);
    *(pNode->fNNiM1) += signN * alphaN * coeffL * pLEdge->dJnDn;
    *(pNode->fNPiM1) += signP * alphaP * coeffL * pLEdge->dJpDp;

    *(pNode->fPPsiiM1) -=
	signN * (-alphaN * coeffL * dJnDpsiPrev +
	coeffL * pLEdge->jn * dAlphaNDpsiM1) +
	signP * (-alphaP * coeffL * dJpDpsiPrev +
	coeffL * pLEdge->jp * dAlphaPDpsiM1);
    *(pNode->fPPiM1) -= signP * alphaP * coeffL * pLEdge->dJpDp;
    *(pNode->fPNiM1) -= signN * alphaN * coeffL * pLEdge->dJnDn;
  }
  if (pNode->nodeI != numNodes - 1) {
    *(pNode->fNPsiiP1) +=
	signN * (alphaN * coeffR * pREdge->dJnDpsiP1 +
	coeffR * pREdge->jn * dAlphaNDpsiP1) +
	signP * (alphaP * coeffR * pREdge->dJpDpsiP1 +
	coeffR * pREdge->jp * dAlphaPDpsiP1);
    *(pNode->fNNiP1) += signN * alphaN * coeffR * pREdge->dJnDnP1;
    *(pNode->fNPiP1) += signP * alphaP * coeffR * pREdge->dJpDpP1;
    *(pNode->fPPsiiP1) -=
	signN * (alphaN * coeffR * pREdge->dJnDpsiP1 +
	coeffR * pREdge->jn * dAlphaNDpsiP1) +
	signP * (alphaP * coeffR * pREdge->dJpDpsiP1 +
	coeffR * pREdge->jp * dAlphaPDpsiP1);
    *(pNode->fPPiP1) -= signP * alphaP * coeffR * pREdge->dJpDpP1;
    *(pNode->fPNiP1) -= signN * alphaN * coeffR * pREdge->dJnDnP1;
  }
  *(pNode->fNPsi) +=
      signN * (alphaN * (-coeffR * pREdge->dJnDpsiP1 +
	  coeffL * dJnDpsiPrev) + (coeffR * pREdge->jn +
	  coeffL * pLEdge->jn) * dAlphaNDpsi) +
      signP * (alphaP * (-coeffR * pREdge->dJpDpsiP1 +
	  coeffL * dJpDpsiPrev) + (coeffR * pREdge->jp +
	  coeffL * pLEdge->jp) * dAlphaPDpsi);
  *(pNode->fNN) += signN * alphaN * (coeffR * pREdge->dJnDn +
      coeffL * pLEdge->dJnDnP1);
  *(pNode->fNP) += signP * alphaP * (coeffR * pREdge->dJpDp +
      coeffL * pLEdge->dJpDpP1);

  *(pNode->fPPsi) -=
      signN * (alphaN * (-coeffR * pREdge->dJnDpsiP1 +
	  coeffL * dJnDpsiPrev) + (coeffR * pREdge->jn +
	  coeffL * pLEdge->jn) * dAlphaNDpsi) +
      signP * (alphaP * (-coeffR * pREdge->dJpDpsiP1 +
	  coeffL * dJpDpsiPrev) + (coeffR * pREdge->jp +
	  coeffL * pLEdge->jp) * dAlphaPDpsi);
  *(pNode->fPN) -= signN * alphaN * (coeffR * pREdge->dJnDn +
      coeffL * pLEdge->dJnDnP1);
  *(pNode->fPP) -= signP * alphaP * (coeffR * pREdge->dJpDp +
      coeffL * pLEdge->dJpDpP1);

  return (generation);
}
