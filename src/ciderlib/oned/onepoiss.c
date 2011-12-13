/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/spmatrix.h"

/* Functions to setup and solve the 1D poisson equation. */


void
ONEQjacBuild(ONEdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  ONEelem *pElem;
  ONEnode *pNode, *pNode1;
  int index;


  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    pNode = pElem->pLeftNode;
    pNode->fPsiPsi = spGetElement(matrix, pNode->poiEqn, pNode->poiEqn);
    pNode1 = pElem->pRightNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn);
    pNode = pElem->pRightNode;
    pNode->fPsiPsi = spGetElement(matrix, pNode->poiEqn, pNode->poiEqn);
    pNode1 = pElem->pLeftNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn);
  }
}


void
ONEQsysLoad(ONEdevice *pDevice)
{
  ONEelem *pElem;
  ONEnode *pNode;
  int index, i;
  double *pRhs = pDevice->rhs;
  double rDx, dPsi;
  double netConc, dNetConc;
  double fNd, fdNd, fNa, fdNa;


  ONEQcommonTerms(pDevice);

  /* zero the rhs vector */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pRhs[index] = 0.0;
  }

  /* zero the matrix */
  spClear(pDevice->matrix);

  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    rDx = pElem->epsRel * pElem->rDx;
    for (i = 0; i <= 1; i++) {
      pNode = pElem->pNodes[i];
      if (pNode->nodeType != CONTACT) {
	*(pNode->fPsiPsi) += rDx;
	pRhs[pNode->poiEqn] += pNode->qf;
	if (pElem->elemType == SEMICON) {
	  netConc = pNode->netConc;
	  dNetConc = 0.0;
	  if (FreezeOut) {
	    ONEQfreezeOut(pNode, &fNd, &fNa, &fdNd, &fdNa);
	    netConc = pNode->nd * fNd - pNode->na * fNa;
	    dNetConc = pNode->nd * fdNd - pNode->na * fdNa;
	  }
	  *(pNode->fPsiPsi) += 0.5 * pElem->dx *
	      (pNode->nConc + pNode->pConc - dNetConc);
	  pRhs[pNode->poiEqn] += 0.5 * pElem->dx *
	      (netConc + pNode->pConc - pNode->nConc);
	}
      }
    }

    dPsi = pElem->pEdge->dPsi;

    pNode = pElem->pLeftNode;
    pRhs[pNode->poiEqn] += rDx * dPsi;
    *(pNode->fPsiPsiiP1) -= rDx;

    pNode = pElem->pRightNode;
    pRhs[pNode->poiEqn] -= rDx * dPsi;
    *(pNode->fPsiPsiiM1) -= rDx;
  }
}


void
ONEQrhsLoad(ONEdevice *pDevice)
{
  ONEelem *pElem;
  ONEnode *pNode;
  int index, i;
  double *pRhs = pDevice->rhs;
  double rDx, dPsi;
  double fNd, fNa, fdNd, fdNa;
  double netConc;


  ONEQcommonTerms(pDevice);

  /* zero the rhs vector */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pRhs[index] = 0.0;
  }

  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    rDx = pElem->epsRel * pElem->rDx;
    for (i = 0; i <= 1; i++) {
      pNode = pElem->pNodes[i];
      if (pNode->nodeType != CONTACT) {
	pRhs[pNode->poiEqn] += pNode->qf;
	if (pElem->elemType == SEMICON) {
	  netConc = pNode->netConc;
	  if (FreezeOut) {
	    ONEQfreezeOut(pNode, &fNd, &fNa, &fdNd, &fdNa);
	    netConc = pNode->nd * fNd - pNode->na * fNa;
	  }
	  pRhs[pNode->poiEqn] += 0.5 * pElem->dx *
	      (netConc + pNode->pConc - pNode->nConc);
	}
      }
    }

    dPsi = pElem->pEdge->dPsi;

    pNode = pElem->pLeftNode;
    pRhs[pNode->poiEqn] += rDx * dPsi;

    pNode = pElem->pRightNode;
    pRhs[pNode->poiEqn] -= rDx * dPsi;
  }
}


void
ONEQcommonTerms(ONEdevice *pDevice)
{
  ONEelem *pElem;
  ONEedge *pEdge;
  ONEnode *pNode;
  int i, index;
  double psi1, psi2, refPsi;
  

  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    refPsi = pElem->matlInfo->refPsi;
    for (i = 0; i <= 1; i++) {
      if (pElem->evalNodes[i]) {
	pNode = pElem->pNodes[i];
	if (pNode->nodeType != CONTACT) {
	  pNode->psi = pDevice->dcSolution[pNode->poiEqn];
	  if (pElem->elemType == SEMICON) {
	    pNode->nConc = pNode->nie * exp(pNode->psi - refPsi);
	    pNode->pConc = pNode->nie * exp(-pNode->psi + refPsi);
	  }
	}
      }
    }
    pEdge = pElem->pEdge;
    pNode = pElem->pNodes[0];
    if (pNode->nodeType != CONTACT) {
      psi1 = pDevice->dcSolution[pNode->poiEqn];
    } else {
      psi1 = pNode->psi;
    }
    pNode = pElem->pNodes[1];
    if (pNode->nodeType != CONTACT) {
      psi2 = pDevice->dcSolution[pNode->poiEqn];
    } else {
      psi2 = pNode->psi;
    }
    pEdge->dPsi = psi2 - psi1;
  }
}
