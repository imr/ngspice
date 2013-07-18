/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/* Functions to compute device conductances and currents */

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/macros.h"
#include "ngspice/spmatrix.h"

#include "onedext.h"
#include "oneddefs.h"

void
NUMDconductance(ONEdevice *pDevice, BOOLEAN tranAnalysis, 
                double *intCoeff, double *gd)
{
  ONEelem *pElem = pDevice->elemArray[pDevice->numNodes - 1];
  ONEnode *pNode;
  ONEedge *pEdge;
  int index;
  double dPsiDv, dNDv, dPDv, *incVpn;


  *gd = 0.0;

  /* zero the rhs before loading in the new rhs */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->rhs[index] = 0.0;
  }
  /* compute incremental changes due to N contact */
  pNode = pElem->pLeftNode;
  pDevice->rhs[pNode->psiEqn] = pElem->epsRel * pElem->rDx;
  if (pElem->elemType == SEMICON) {
    pEdge = pElem->pEdge;
    pDevice->rhs[pNode->nEqn] = -pEdge->dJnDpsiP1;
    pDevice->rhs[pNode->pEqn] = -pEdge->dJpDpsiP1;
  }
  incVpn = pDevice->dcDeltaSolution;
  spSolve(pDevice->matrix, pDevice->rhs, incVpn, NULL, NULL);

  pElem = pDevice->elemArray[1];
  pNode = pElem->pRightNode;
  pEdge = pElem->pEdge;
  dPsiDv = incVpn[pNode->psiEqn];
  if (pElem->elemType == SEMICON) {
    dNDv = incVpn[pNode->nEqn];
    dPDv = incVpn[pNode->pEqn];
    *gd += pEdge->dJnDpsiP1 * dPsiDv + pEdge->dJnDnP1 * dNDv +
	pEdge->dJpDpsiP1 * dPsiDv + pEdge->dJpDpP1 * dPDv;
  }
  /* For transient analysis, add the displacement term */
  if (tranAnalysis) {
    *gd -= intCoeff[0] * pElem->epsRel * pElem->rDx * dPsiDv;
  }
  *gd *= -GNorm * pDevice->area;
  
}

void
NBJTconductance(ONEdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff,
    double *dIeDVce, double *dIcDVce, double *dIeDVbe, double *dIcDVbe)
{
  ONEelem *pLastElem = pDevice->elemArray[pDevice->numNodes - 1];
  ONEelem *pBaseElem = pDevice->elemArray[pDevice->baseIndex - 1];
  ONEelem *pElem;
  ONEnode *pNode;
  ONEedge *pEdge;
  int index;
  double dPsiDVce, dPsiDVbe, dNDVce, dNDVbe, dPDVce, dPDVbe;
  double *incVce, *incVbe;
  double nConc, pConc;
  double area = pDevice->area;

  *dIeDVce = 0.0;
  *dIcDVce = 0.0;
  *dIeDVbe = 0.0;
  *dIcDVbe = 0.0;

  /* zero the rhs before loading in the new rhs */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->rhs[index] = 0.0;
  }
  /* store the new rhs for computing CE incremental quantities */
  pNode = pLastElem->pLeftNode;
  pDevice->rhs[pNode->psiEqn] = pLastElem->epsRel * pLastElem->rDx;
  if (pLastElem->elemType == SEMICON) {
    pEdge = pLastElem->pEdge;
    pDevice->rhs[pNode->nEqn] = -pEdge->dJnDpsiP1;
    pDevice->rhs[pNode->pEqn] = -pEdge->dJpDpsiP1;
  }
  incVce = pDevice->dcDeltaSolution;
  spSolve(pDevice->matrix, pDevice->rhs, incVce, NULL, NULL);

  /* zero the rhs before loading in the new rhs base contribution */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->rhs[index] = 0.0;
  }
  pNode = pBaseElem->pRightNode;

  if (pNode->baseType == N_TYPE) {
    nConc = pDevice->devState0 [pNode->nodeN];
    pDevice->rhs[pNode->nEqn] = nConc * pNode->eg;
  } else if (pNode->baseType == P_TYPE) {
    pConc = pDevice->devState0 [pNode->nodeP];
    pDevice->rhs[pNode->pEqn] = pConc * pNode->eg;
  } else {
    printf("NBJTconductance: unknown base type\n");
  }

  incVbe = pDevice->copiedSolution;
  spSolve(pDevice->matrix, pDevice->rhs, incVbe, NULL, NULL);

  pElem = pDevice->elemArray[1];/* first element */
  pEdge = pElem->pEdge;
  pNode = pElem->pRightNode;

  dPsiDVce = incVce[pNode->psiEqn];
  dPsiDVbe = incVbe[pNode->psiEqn];
  if (pElem->elemType == SEMICON) {
    dNDVce = incVce[pNode->nEqn];
    dPDVce = incVce[pNode->pEqn];
    dNDVbe = incVbe[pNode->nEqn];
    dPDVbe = incVbe[pNode->pEqn];
    *dIeDVce += pEdge->dJnDpsiP1 * dPsiDVce + pEdge->dJnDnP1 * dNDVce +
	pEdge->dJpDpsiP1 * dPsiDVce + pEdge->dJpDpP1 * dPDVce;
    *dIeDVbe += pEdge->dJnDpsiP1 * dPsiDVbe + pEdge->dJnDnP1 * dNDVbe +
	pEdge->dJpDpsiP1 * dPsiDVbe + pEdge->dJpDpP1 * dPDVbe;
  }
  /* For transient analysis add the displacement term */
  if (tranAnalysis) {
    *dIeDVce -= intCoeff[0] * pElem->epsRel * dPsiDVce * pElem->rDx;
    *dIeDVbe -= intCoeff[0] * pElem->epsRel * dPsiDVbe * pElem->rDx;
  }
  pElem = pDevice->elemArray[pDevice->numNodes - 1];	/* last element */
  pEdge = pElem->pEdge;
  pNode = pElem->pLeftNode;
  dPsiDVce = incVce[pNode->psiEqn];
  dPsiDVbe = incVbe[pNode->psiEqn];
  if (pElem->elemType == SEMICON) {
    dNDVce = incVce[pNode->nEqn];
    dPDVce = incVce[pNode->pEqn];
    dNDVbe = incVbe[pNode->nEqn];
    dPDVbe = incVbe[pNode->pEqn];
    *dIcDVce += -pEdge->dJnDpsiP1 * dPsiDVce + pEdge->dJnDn * dNDVce +
	-pEdge->dJpDpsiP1 * dPsiDVce + pEdge->dJpDp * dPDVce +
    /* add terms since adjacent to boundary */
	pEdge->dJnDpsiP1 + pEdge->dJpDpsiP1;
    *dIcDVbe += -pEdge->dJnDpsiP1 * dPsiDVbe + pEdge->dJnDn * dNDVbe +
	-pEdge->dJpDpsiP1 * dPsiDVbe + pEdge->dJpDp * dPDVbe;
  }
  if (tranAnalysis) {
    *dIcDVce += intCoeff[0] * pElem->epsRel * (dPsiDVce - 1.0) * pElem->rDx;
    *dIcDVbe += intCoeff[0] * pElem->epsRel * dPsiDVbe * pElem->rDx;
  }
  *dIeDVce *= -GNorm * area;
  *dIcDVce *= -GNorm * area;
  *dIeDVbe *= -GNorm * area;
  *dIcDVbe *= -GNorm * area;
}

void
NUMDcurrent(ONEdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff, 
            double *id)
{
  ONEnode *pNode;
  ONEelem *pElem;
  ONEedge *pEdge;
  double *delta = pDevice->dcDeltaSolution;
  double dPsi, dN, dP;

  *id = 0.0;

  pElem = pDevice->elemArray[1];
  pNode = pElem->pRightNode;
  pEdge = pElem->pEdge;
  dPsi = delta[pNode->psiEqn];
  *id = pEdge->jn + pEdge->jp + pElem->epsRel * pEdge->jd;
  if (pElem->elemType == SEMICON) {
    dN = delta[pNode->nEqn];
    dP = delta[pNode->pEqn];
    *id += pEdge->dJnDpsiP1 * dPsi + pEdge->dJnDnP1 * dN +
	pEdge->dJpDpsiP1 * dPsi + pEdge->dJpDpP1 * dP;    
  }
  /* for transient analysis add the displacement term */
  if (tranAnalysis) {
    *id -= intCoeff[0] * pElem->epsRel * pElem->rDx * dPsi;
  }
  *id *= JNorm * pDevice->area;
}

void
NBJTcurrent(ONEdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff, 
            double *ie, double *ic)
{
  ONEnode *pNode;
  ONEelem *pElem;
  ONEedge *pEdge;
  double dPsi, dN, dP;
  double *solution;

  solution = pDevice->dcDeltaSolution;

  /* first edge for calculating ie */
  pElem = pDevice->elemArray[1];
  pNode = pElem->pRightNode;
  pEdge = pElem->pEdge;
  dPsi = solution[pNode->psiEqn];
  *ie = pEdge->jn + pEdge->jp + pElem->epsRel * pEdge->jd;
  if (pElem->elemType == SEMICON) {
    dN = solution[pNode->nEqn];
    dP = solution[pNode->pEqn];
    *ie += pEdge->dJnDpsiP1 * dPsi + pEdge->dJnDnP1 * dN +
	pEdge->dJpDpsiP1 * dPsi + pEdge->dJpDpP1 * dP;
  }
  /* for transient analysis add the displacement term */
  if (tranAnalysis) {
    *ie -= intCoeff[0] * pElem->epsRel * dPsi * pElem->rDx;
  }
  /* last edge for calculating ic */
  pElem = pDevice->elemArray[pDevice->numNodes - 1];
  pNode = pElem->pLeftNode;
  pEdge = pElem->pEdge;
  dPsi = solution[pNode->psiEqn];
  *ic = pEdge->jn + pEdge->jp + pElem->epsRel * pEdge->jd;
  if (pElem->elemType == SEMICON) {
    dN = solution[pNode->nEqn];
    dP = solution[pNode->pEqn];
    *ic += -pEdge->dJnDpsiP1 * dPsi + pEdge->dJnDn * dN +
	-pEdge->dJpDpsiP1 * dPsi + pEdge->dJpDp * dP;
  }
  if (tranAnalysis) {
    *ic += intCoeff[0] * pElem->epsRel * dPsi * pElem->rDx;
  }
  *ic *= -JNorm * pDevice->area;
  *ie *= -JNorm * pDevice->area;
}
