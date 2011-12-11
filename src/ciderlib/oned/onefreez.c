/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/onemesh.h"
#include "../../maths/misc/accuracy.h"
#include "onedext.h"
#include "oneddefs.h"


void
ONEQfreezeOut(ONEnode *pNode, double *ndFac, double *naFac, double *dNdFac,
              double *dNaFac)
{
  double temp1, temp2;
  double eLev;
  ONEmaterial *info;


  if (pNode->pRightElem && pNode->pRightElem->evalNodes[0]) {
    info = pNode->pRightElem->matlInfo;
  } else {
    info = pNode->pLeftElem->matlInfo;
  }

  eLev = info->eDon;
  if (info->material != GAAS) {
    eLev -= LEVEL_ALPHA_SI * pow(pNode->nd * NNorm, 1.0 / 3.0);
    if (eLev < 0.0)
      eLev = 0.0;
  }
  if (eLev >= ExpLim) {
    *ndFac = 0.0;
    *dNdFac = 0.0;
  } else if (eLev <= -ExpLim) {
    *ndFac = 1.0;
    *dNdFac = 0.0;
  } else {
    temp1 = info->gDon * pNode->nConc * NNorm * exp(eLev) / info->nc0;
    temp2 = 1.0 / (1.0 + temp1);
    *ndFac = temp2;
    *dNdFac = -temp2 * temp2 * temp1;
  }

  eLev = info->eAcc;
  if (info->material != GAAS) {
    eLev -= LEVEL_ALPHA_SI * pow(pNode->na * NNorm, 1.0 / 3.0);
    if (eLev < 0.0)
      eLev = 0.0;
  }
  if (eLev >= ExpLim) {
    *naFac = 0.0;
    *dNaFac = 0.0;
  } else if (eLev <= -ExpLim) {
    *naFac = 1.0;
    *dNaFac = 0.0;
  } else {
    temp1 = info->gAcc * pNode->pConc * NNorm * exp(eLev) / info->nv0;
    temp2 = 1.0 / (1.0 + temp1);
    *naFac = temp2;
    *dNaFac = temp2 * temp2 * temp1;
  }
}

void
ONE_freezeOut(ONEnode *pNode, double nConc, double pConc, double *ndFac, 
              double *naFac, double *dNdFac, double *dNaFac)
{
  double temp1, temp2;
  double eLev;
  ONEmaterial *info;


  if (pNode->pRightElem && pNode->pRightElem->evalNodes[0]) {
    info = pNode->pRightElem->matlInfo;
  } else {
    info = pNode->pLeftElem->matlInfo;
  }

  eLev = info->eDon;
  if (info->material != GAAS) {
    eLev -= LEVEL_ALPHA_SI * pow(pNode->nd * NNorm, 1.0 / 3.0);
    if (eLev < 0.0)
      eLev = 0.0;
  }
  if (eLev >= ExpLim) {
    *ndFac = 0.0;
    *dNdFac = 0.0;
  } else if (eLev <= -ExpLim) {
    *ndFac = 1.0;
    *dNdFac = 0.0;
  } else {
    temp1 = info->gDon * exp(eLev) * NNorm / info->nc0;
    temp2 = 1.0 / (1.0 + nConc * temp1);
    *ndFac = temp2;
    *dNdFac = -temp2 * temp2 * temp1;
  }

  eLev = info->eAcc;
  if (info->material != GAAS) {
    eLev -= LEVEL_ALPHA_SI * pow(pNode->na * NNorm, 1.0 / 3.0);
    if (eLev < 0.0)
      eLev = 0.0;
  }
  if (eLev >= ExpLim) {
    *naFac = 0.0;
    *dNaFac = 0.0;
  } else if (eLev <= -ExpLim) {
    *naFac = 1.0;
    *dNaFac = 0.0;
  } else {
    temp1 = info->gAcc * exp(eLev) * NNorm / info->nv0;
    temp2 = 1.0 / (1.0 + pConc * temp1);
    *naFac = temp2;
    *dNaFac = -temp2 * temp2 * temp1;
  }
}
