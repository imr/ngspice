/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/spmatrix.h"
#include "ngspice/macros.h"
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/cidersupt.h"
#include "../../maths/misc/bernoull.h"


/* functions to setup and solve the continuity equations */
/* Both continuity equations are solved */


void 
ONE_jacBuild(ONEdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  ONEelem *pElem;
  ONEnode *pNode;
  int index, eIndex;
  int psiEqn, nEqn, pEqn;	/* scratch for deref'd eqn numbers */
  int psiEqnL=0, nEqnL=0, pEqnL=0;
  int psiEqnR=0, nEqnR=0, pEqnR=0;


  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    /* first the self terms */
    for (index = 0; index <= 1; index++) {
      pNode = pElem->pNodes[index];
      /* get poisson pointer */
      psiEqn = pNode->psiEqn;
      pNode->fPsiPsi = spGetElement(matrix, psiEqn, psiEqn);

      if (pElem->elemType == SEMICON) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pEqn = pNode->pEqn;
	/* pointers for additional terms */
	pNode->fPsiN = spGetElement(matrix, psiEqn, nEqn);
	pNode->fPsiP = spGetElement(matrix, psiEqn, pEqn);
	pNode->fNPsi = spGetElement(matrix, nEqn, psiEqn);
	pNode->fNN = spGetElement(matrix, nEqn, nEqn);
	pNode->fNP = spGetElement(matrix, nEqn, pEqn);
	pNode->fPPsi = spGetElement(matrix, pEqn, psiEqn);
	pNode->fPP = spGetElement(matrix, pEqn, pEqn);
	pNode->fPN = spGetElement(matrix, pEqn, nEqn);
      } else {
	nEqn = 0;
	pEqn = 0;
      }

      /* save indices */
      if (index == 0) {		/* left node */
	psiEqnL = psiEqn;
	nEqnL = nEqn;
	pEqnL = pEqn;
      } else {
	psiEqnR = psiEqn;
	nEqnR = nEqn;
	pEqnR = pEqn;
      }
    }

    /* now terms to couple to adjacent nodes */
    pNode = pElem->pLeftNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, psiEqnL, psiEqnR);
    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */
      pNode->fNPsiiP1 = spGetElement(matrix, nEqnL, psiEqnR);
      pNode->fNNiP1 = spGetElement(matrix, nEqnL, nEqnR);
      pNode->fPPsiiP1 = spGetElement(matrix, pEqnL, psiEqnR);
      pNode->fPPiP1 = spGetElement(matrix, pEqnL, pEqnR);
      if (AvalancheGen) {
	pNode->fNPiP1 = spGetElement(matrix, nEqnL, pEqnR);
	pNode->fPNiP1 = spGetElement(matrix, pEqnL, nEqnR);
      }
    }
    pNode = pElem->pRightNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, psiEqnR, psiEqnL);
    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */
      pNode->fNPsiiM1 = spGetElement(matrix, nEqnR, psiEqnL);
      pNode->fNNiM1 = spGetElement(matrix, nEqnR, nEqnL);
      pNode->fPPsiiM1 = spGetElement(matrix, pEqnR, psiEqnL);
      pNode->fPPiM1 = spGetElement(matrix, pEqnR, pEqnL);
      if (AvalancheGen) {
	pNode->fNPiM1 = spGetElement(matrix, nEqnR, pEqnL);
	pNode->fPNiM1 = spGetElement(matrix, pEqnR, nEqnL);
      }
    }
  }
}


void 
ONE_sysLoad(ONEdevice *pDevice, BOOLEAN tranAnalysis, 
            ONEtranInfo *info)
{
  ONEelem *pElem;
  ONEnode *pNode;
  ONEedge *pEdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dx, rDx, dPsi;
  double generation;
  double perTime = 0.0;
  double fNd, fNa, fdNd, fdNa;
  double netConc, dNd, dNa, psi, nConc, pConc;


  /* first compute the currents and their derivatives */
  ONE_commonTerms(pDevice, FALSE, tranAnalysis, info);

  /* find reciprocal timestep */
  if (tranAnalysis) {
    perTime = info->intCoeff[0];
  }
  /* zero the rhs vector */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pRhs[index] = 0.0;
  }

  /* zero the matrix */
  spClear(pDevice->matrix);

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    dx = 0.5 * pElem->dx;
    rDx = pElem->epsRel * pElem->rDx;

    /* load for all i */
    for (index = 0; index <= 1; index++) {
      pNode = pElem->pNodes[index];
      if (pNode->nodeType != CONTACT) {
	*(pNode->fPsiPsi) += rDx;
	pRhs[pNode->psiEqn] += pNode->qf;
	if (pElem->elemType == SEMICON) {
	  pEdge = pElem->pEdge;
	  netConc = pNode->netConc;
	  dNd = 0.0;
	  dNa = 0.0;
	  psi = pDevice->devState0 [pNode->nodePsi];
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];
	  
	  
	  if (FreezeOut) {
	    ONE_freezeOut(pNode, nConc, pConc, &fNd, &fNa, &fdNd, &fdNa);
	    netConc = pNode->nd * fNd - pNode->na * fNa;
	    dNd = pNode->nd * fdNd;
	    dNa = pNode->na * fdNa;
	  }
	  *(pNode->fPsiN) += dx * (1.0 - dNd);
	  *(pNode->fPsiP) -= dx * (1.0 - dNa);
	  *(pNode->fNPsi) -= pEdge->dJnDpsiP1;
	  *(pNode->fPPsi) -= pEdge->dJpDpsiP1;

	  pRhs[pNode->psiEqn] += dx * (netConc + pConc - nConc);

	  /* Handle generation terms */
	  *(pNode->fNN) -= dx * pNode->dUdN;
	  *(pNode->fNP) -= dx * pNode->dUdP;
	  *(pNode->fPP) += dx * pNode->dUdP;
	  *(pNode->fPN) += dx * pNode->dUdN;
	  pRhs[pNode->nEqn] -= -dx * pNode->uNet;
	  pRhs[pNode->pEqn] -= dx * pNode->uNet;

	  /* Handle dXdT continuity terms */
	  if (tranAnalysis) {
	    *(pNode->fNN) -= dx * perTime;
	    *(pNode->fPP) += dx * perTime;
	    pRhs[pNode->nEqn] += dx * pNode->dNdT;
	    pRhs[pNode->pEqn] -= dx * pNode->dPdT;
	  }
	  /* Take care of base contact if necessary */
	  /* eg holds the base edge mu/dx */
	  if (pNode->baseType == N_TYPE) {
	    pRhs[pNode->nEqn] += 0.5 * pNode->eg * nConc *
		(pNode->vbe - psi + log(nConc / pNode->nie));
	    *(pNode->fNPsi) += 0.5 * pNode->eg * nConc;
	    *(pNode->fNN) -= 0.5 * pNode->eg *
		(pNode->vbe - psi + log(nConc / pNode->nie) + 1.0);
	  } else if (pNode->baseType == P_TYPE) {
	    pRhs[pNode->pEqn] += 0.5 * pNode->eg * pConc *
		(pNode->vbe - psi - log(pConc / pNode->nie));
	    *(pNode->fPPsi) += 0.5 * pNode->eg * pConc;
	    *(pNode->fPP) -= 0.5 * pNode->eg *
		(pNode->vbe - psi - log(pConc / pNode->nie) - 1.0);
	  }
	}
      }
    }

    pEdge = pElem->pEdge;
    dPsi = pEdge->dPsi;

    pNode = pElem->pLeftNode;
    if (pNode->nodeType != CONTACT) {
      pRhs[pNode->psiEqn] += rDx * dPsi;
      *(pNode->fPsiPsiiP1) -= rDx;
      if (pElem->elemType == SEMICON) {
	pRhs[pNode->nEqn] -= pEdge->jn;
	pRhs[pNode->pEqn] -= pEdge->jp;
	
	*(pNode->fNN) += pEdge->dJnDn;
	*(pNode->fPP) += pEdge->dJpDp;
	*(pNode->fNPsiiP1) += pEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += pEdge->dJnDnP1;
	*(pNode->fPPsiiP1) += pEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += pEdge->dJpDpP1;
      }
    }
    pNode = pElem->pRightNode;
    if (pNode->nodeType != CONTACT) {
      pRhs[pNode->psiEqn] -= rDx * dPsi;
      *(pNode->fPsiPsiiM1) -= rDx;
      if (pElem->elemType == SEMICON) {
	pRhs[pNode->nEqn] += pEdge->jn;
	pRhs[pNode->pEqn] += pEdge->jp;
	
	
	*(pNode->fNN) -= pEdge->dJnDnP1;
	*(pNode->fPP) -= pEdge->dJpDpP1;
	*(pNode->fNPsiiM1) += pEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= pEdge->dJnDn;
	*(pNode->fPPsiiM1) += pEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= pEdge->dJpDp;
      }
    }
  }
  if (AvalancheGen) {
    /* add the generation terms */
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if ((pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON)) {
	    generation = ONEavalanche(FALSE, pDevice, pNode);
	    pRhs[pNode->nEqn] -= generation;
	    pRhs[pNode->pEqn] += generation;
	  }
	}
      }
    }
  }
}


void 
ONE_jacLoad(ONEdevice *pDevice)
{
  /* used only for ac analysis */
  ONEelem *pElem;
  ONEnode *pNode, *pNode1;
  ONEedge *pEdge;
  int index, eIndex;
  double dx, rDx, dPsi;
  double generation;
  double fNd, fNa, fdNd, fdNa;
  double dNd, dNa, psi, nConc, pConc;


  /* first compute the currents and their derivatives */
  ONE_commonTerms(pDevice, FALSE, FALSE, NULL);

  /* zero the matrix */
  spClear(pDevice->matrix);

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    dx = 0.5 * pElem->dx;
    rDx = pElem->epsRel * pElem->rDx;
    /* load for all i */
    for (index = 0; index <= 1; index++) {
      pNode = pElem->pNodes[index];
      if (pNode->nodeType != CONTACT) {
	*(pNode->fPsiPsi) += rDx;
	if (pElem->elemType == SEMICON) {
	  pEdge = pElem->pEdge;
	  dNd = 0.0;
	  dNa = 0.0;
	  psi = pDevice->devState0 [pNode->nodePsi];
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];
	  if (FreezeOut) {
	    ONE_freezeOut(pNode, nConc, pConc, &fNd, &fNa, &fdNd, &fdNa);
	    dNd = pNode->nd * fdNd;
	    dNa = pNode->na * fdNa;
	  }
	  *(pNode->fPsiN) += dx * (1.0 - dNd);
	  *(pNode->fPsiP) -= dx * (1.0 - dNa);
	  *(pNode->fNPsi) -= pEdge->dJnDpsiP1;
	  *(pNode->fPPsi) -= pEdge->dJpDpsiP1;

	  if (pNode->baseType == N_TYPE) {
	    *(pNode->fNPsi) += 0.5 * nConc * pNode->eg;
	    *(pNode->fNN) -= 0.5 * pNode->eg
		* (pNode->vbe - psi + log(nConc / pNode->nie) + 1.0);
	  }
	  if (pNode->baseType == P_TYPE) {
	    *(pNode->fPPsi) += 0.5 * pConc * pNode->eg;
	    *(pNode->fPP) -= 0.5 * pNode->eg
		* (pNode->vbe - psi - log(pConc / pNode->nie) - 1.0);
	  }
	}
      }
    }
    pNode = pElem->pLeftNode;
    if (pNode->nodeType != CONTACT) {
      pEdge = pElem->pEdge;
      dPsi = pEdge->dPsi;
      if (pElem->elemType == SEMICON) {
	*(pNode->fNN) += pEdge->dJnDn - dx * pNode->dUdN;
	*(pNode->fNP) -= dx * pNode->dUdP;
	*(pNode->fPP) += pEdge->dJpDp + dx * pNode->dUdP;
	*(pNode->fPN) += dx * pNode->dUdN;
      }
      pNode1 = pElem->pRightNode;
      if (pNode1->nodeType != CONTACT) {
	*(pNode->fPsiPsiiP1) -= rDx;
	if (pElem->elemType == SEMICON) {
	  *(pNode->fNPsiiP1) += pEdge->dJnDpsiP1;
	  *(pNode->fNNiP1) += pEdge->dJnDnP1;
	  *(pNode->fPPsiiP1) += pEdge->dJpDpsiP1;
	  *(pNode->fPPiP1) += pEdge->dJpDpP1;
	}
      }
    }
    pNode = pElem->pRightNode;
    if (pNode->nodeType != CONTACT) {
      pEdge = pElem->pEdge;
      dPsi = pEdge->dPsi;
      if (pElem->elemType == SEMICON) {
	*(pNode->fNN) += -pEdge->dJnDnP1 - dx * pNode->dUdN;
	*(pNode->fNP) -= dx * pNode->dUdP;
	*(pNode->fPP) += -pEdge->dJpDpP1 + dx * pNode->dUdP;
	*(pNode->fPN) += dx * pNode->dUdN;
      }
      pNode1 = pElem->pLeftNode;
      if (pNode1->nodeType != CONTACT) {
	*(pNode->fPsiPsiiM1) -= rDx;
	if (pElem->elemType == SEMICON) {
	  *(pNode->fNPsiiM1) += pEdge->dJnDpsiP1;
	  *(pNode->fNNiM1) -= pEdge->dJnDn;
	  *(pNode->fPPsiiM1) += pEdge->dJpDpsiP1;
	  *(pNode->fPPiM1) -= pEdge->dJpDp;
	}
      }
    }
  }
  if (AvalancheGen) {
    /* add the generation terms */
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if ((pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON)) {
	    generation = ONEavalanche(FALSE, pDevice, pNode);
	  }
	}
      }
    }
  }
}

void 
ONE_rhsLoad(ONEdevice *pDevice, BOOLEAN tranAnalysis,
            ONEtranInfo *info)
{
  ONEelem *pElem;
  ONEnode *pNode;
  ONEedge *pEdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dx, rDx, dPsi;
  double generation;
  double perTime;
  double fNd, fNa, fdNd, fdNa;
  double netConc, dNd, dNa, psi, nConc, pConc;

  /* first compute the currents and their derivatives */
  ONE_commonTerms(pDevice, FALSE, tranAnalysis, info);

  /* find reciprocal timestep */
  if (tranAnalysis) {
    perTime = info->intCoeff[0];
  }
  /* zero the rhs vector */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pRhs[index] = 0.0;
  }

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];

    dx = 0.5 * pElem->dx;
    rDx = pElem->epsRel * pElem->rDx;

    /* load for all i */
    for (index = 0; index <= 1; index++) {
      pNode = pElem->pNodes[index];
      if (pNode->nodeType != CONTACT) {
	pRhs[pNode->psiEqn] += pNode->qf;
	if (pElem->elemType == SEMICON) {
	  pEdge = pElem->pEdge;
	  netConc = pNode->netConc;
	  dNd = 0.0;
	  dNa = 0.0;
	  psi = pDevice->devState0 [pNode->nodePsi];
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];
	  if (FreezeOut) {
	    ONE_freezeOut(pNode, nConc, pConc, &fNd, &fNa, &fdNd, &fdNa);
	    netConc = pNode->nd * fNd - pNode->na * fNa;
	    dNd = pNode->nd * fdNd;
	    dNa = pNode->na * fdNa;
	  }
	  pRhs[pNode->psiEqn] += dx * (netConc + pConc - nConc);

	  /* Handle generation terms */
	  pRhs[pNode->nEqn] -= -dx * pNode->uNet;
	  pRhs[pNode->pEqn] -= dx * pNode->uNet;

	  /* Handle dXdT continuity terms */
	  if (tranAnalysis) {
	    pRhs[pNode->nEqn] += dx * pNode->dNdT;
	    pRhs[pNode->pEqn] -= dx * pNode->dPdT;
	  }
	  /* Take care of base contact if necessary */
	  /* eg holds the base edge mu/dx */
	  if (pNode->baseType == N_TYPE) {
	    pRhs[pNode->nEqn] += 0.5 * pNode->eg * nConc *
		(pNode->vbe - psi + log(nConc / pNode->nie));
	  } else if (pNode->baseType == P_TYPE) {
	    pRhs[pNode->pEqn] += 0.5 * pNode->eg * pConc *
		(pNode->vbe - psi - log(pConc / pNode->nie));
	  }
	}
      }
    }

    pEdge = pElem->pEdge;
    dPsi = pEdge->dPsi;

    pNode = pElem->pLeftNode;
    if (pNode->nodeType != CONTACT) {
      pRhs[pNode->psiEqn] += rDx * dPsi;
      if (pElem->elemType == SEMICON) {
	pRhs[pNode->nEqn] -= pEdge->jn;
	pRhs[pNode->pEqn] -= pEdge->jp;
      }
    }
    pNode = pElem->pRightNode;
    if (pNode->nodeType != CONTACT) {
      pRhs[pNode->psiEqn] -= rDx * dPsi;
      if (pElem->elemType == SEMICON) {
	pRhs[pNode->nEqn] += pEdge->jn;
	pRhs[pNode->pEqn] += pEdge->jp;
      }
    }
  }
  if (AvalancheGen) {
    /* add the generation terms */
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if ((pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON)) {
	    generation = ONEavalanche(TRUE, pDevice, pNode);
	    pRhs[pNode->nEqn] -= generation;
	    pRhs[pNode->pEqn] += generation;
	  }
	}
      }
    }
  }
}

void 
ONE_commonTerms(ONEdevice *pDevice, BOOLEAN currentOnly, 
                BOOLEAN tranAnalysis, ONEtranInfo *info)
{
  ONEelem *pElem;
  ONEedge *pEdge;
  ONEnode *pNode;
  int index, eIndex;
  double psi1, psi2, psi, nConc=0.0, pConc=0.0, nC, pC, nP1, pP1;
  double dPsiN, dPsiP;
  double bPsiN, dbPsiN, bMPsiN, dbMPsiN;
  double bPsiP, dbPsiP, bMPsiP, dbMPsiP;
  double mun, dMun, mup, dMup;
  double conc1, conc2;
  double cnAug, cpAug;


  /* evaluate all node (including recombination) and edge quantities */
  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    cnAug = pElem->matlInfo->cAug[ELEC];
    cpAug = pElem->matlInfo->cAug[HOLE];
    for (index = 0; index <= 1; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	if (pNode->nodeType != CONTACT) {
	  psi = pDevice->dcSolution[pNode->psiEqn];
	  if (pElem->elemType == SEMICON) {
	    nConc = pDevice->dcSolution[pNode->nEqn];
	    pConc = pDevice->dcSolution[pNode->pEqn];
	    if (Srh) {
	      recomb(nConc, pConc,
		  pNode->tn, pNode->tp, cnAug, cpAug, pNode->nie,
		  &pNode->uNet, &pNode->dUdN, &pNode->dUdP);
	    } else {
	      pNode->uNet = 0.0;
	      pNode->dUdN = 0.0;
	      pNode->dUdP = 0.0;
	    }
	    if (pNode->baseType == P_TYPE && pConc <= 0.0) {
	      pConc = pNode->na;
	    } else if (pNode->baseType == N_TYPE && nConc <= 0.0) {
	      nConc = pNode->nd;
	    }
	  }
	} else {
	  /* a contact node */
	  psi = pNode->psi;
	  if (pElem->elemType == SEMICON) {
	    nConc = pNode->nConc;
	    pConc = pNode->pConc;
	  }
	}
	/* store info in the state tables */
	pDevice->devState0 [pNode->nodePsi] = psi;
	if (pElem->elemType == SEMICON) {
	  pDevice->devState0 [pNode->nodeN] = nConc;
	  pDevice->devState0 [pNode->nodeP] = pConc;
	  if (tranAnalysis && pNode->nodeType != CONTACT) {
	    pNode->dNdT = integrate(pDevice->devStates, info, pNode->nodeN);
	    pNode->dPdT = integrate(pDevice->devStates, info, pNode->nodeP);
	  }
	}
      }
    }
    pEdge = pElem->pEdge;
    pNode = pElem->pLeftNode;
    if (pNode->nodeType != CONTACT) {
      psi1 = pDevice->dcSolution[pNode->psiEqn];
    } else {
      psi1 = pNode->psi;
    }
    pNode = pElem->pRightNode;
    if (pNode->nodeType != CONTACT) {
      psi2 = pDevice->dcSolution[pNode->psiEqn];
    } else {
      psi2 = pNode->psi;
    }
    pEdge->dPsi = psi2 - psi1;
    pDevice->devState0 [pEdge->edgeDpsi] = pEdge->dPsi;
  }
  /* calculate the current densities and mobility values */
  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    pEdge = pElem->pEdge;
    if (pElem->elemType == SEMICON) {
      dPsiN = pEdge->dPsi + pEdge->dCBand;
      dPsiP = pEdge->dPsi - pEdge->dVBand;
      bernoulli(dPsiN, &bPsiN, &dbPsiN, &bMPsiN, &dbMPsiN, !currentOnly);
      bernoulli(dPsiP, &bPsiP, &dbPsiP, &bMPsiP, &dbMPsiP, !currentOnly);
      nC = pDevice->devState0 [pElem->pLeftNode->nodeN];
      nP1 = pDevice->devState0 [pElem->pRightNode->nodeN];
      pC = pDevice->devState0 [pElem->pLeftNode->nodeP];
      pP1 = pDevice->devState0 [pElem->pRightNode->nodeP];
      conc1 = pElem->pLeftNode->totalConc;
      conc2 = pElem->pRightNode->totalConc;
      pEdge->jn = (bPsiN * nP1 - bMPsiN * nC);
      pEdge->jp = (bPsiP * pC - bMPsiP * pP1);
      
      
      mun = pEdge->mun;
      dMun = 0.0;
      mup = pEdge->mup;
      dMup = 0.0;
      MOBfieldDep(pElem->matlInfo, ELEC, dPsiN * pElem->rDx, &mun, &dMun);
      MOBfieldDep(pElem->matlInfo, HOLE, dPsiP * pElem->rDx, &mup, &dMup);
      
      
      mun *= pElem->rDx;
      dMun *= pElem->rDx * pElem->rDx;
      mup *= pElem->rDx;
      dMup *= pElem->rDx * pElem->rDx;
      

      
      /*
       * The base continuity equation makes use of mu/dx in eg. The base
       * length has already been calculated and converted to normalized,
       * reciprocal form during setup. The name should be changed, but that's
       * a big hassle.
       */
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->baseType == N_TYPE) {
	    pNode->eg = pEdge->mun * pDevice->baseLength;
	  } else if (pNode->baseType == P_TYPE) {
	    pNode->eg = pEdge->mup * pDevice->baseLength;
	  }
	}
      }

      pEdge->jn *= mun;
      pEdge->jp *= mup;
      
      
      if (!currentOnly) {
	if (dMun == 0.0) {
	  pEdge->dJnDpsiP1 = mun * (dbPsiN * nP1 - dbMPsiN * nC);
	} else {
	  pEdge->dJnDpsiP1 = dMun * (bPsiN * nP1 - bMPsiN * nC)
	      + mun * (dbPsiN * nP1 - dbMPsiN * nC);
	}
	pEdge->dJnDn = -mun * bMPsiN;
	pEdge->dJnDnP1 = mun * bPsiN;
	if (dMup == 0.0) {
	  pEdge->dJpDpsiP1 = mup * (dbPsiP * pC - dbMPsiP * pP1);
	} else {
	  pEdge->dJpDpsiP1 = dMup * (bPsiP * pC - bMPsiP * pP1)
	      + mup * (dbPsiP * pC - dbMPsiP * pP1);
	}
	pEdge->dJpDp = mup * bPsiP;
	pEdge->dJpDpP1 = -mup * bMPsiP;
      }
    }
    if (tranAnalysis) {
      pEdge->jd = -integrate(pDevice->devStates, info,
	  pEdge->edgeDpsi) * pElem->rDx;
    }
  }
}
