/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/**********
One-Dimensional Numerical Device Setup Routines
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/carddefs.h"		/* XXX Not really modular if we need this. */
/* #include "material.h" */
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/cidersupt.h"

/* compute node parameters */
void
ONEsetup(ONEdevice *pDevice)
{
  double temp1, deltaEg, avgConc, totalConc, absNetConc;
  double ncv0, dBand, dNie, psiBand[2];
  int index, eIndex;
  ONEnode *pNode;
  ONEelem *pElem;
  ONEedge *pEdge;
  ONEmaterial *info;


  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    info = pElem->matlInfo;

    pElem->dx = pElem->pRightNode->x - pElem->pLeftNode->x;
    pElem->epsRel = info->eps;

    if (pElem->elemType == INSULATOR) {
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType == CONTACT) {
	    pNode->eaff = PHI_METAL;
	    pNode->eg = 0.0;
	  } else {
	    pNode->eaff = info->affin;
	    pNode->eg = info->eg0;
	  }
	}
      }
    } else if (pElem->elemType == SEMICON) {
      ncv0 = sqrt(info->nc0) * sqrt(info->nv0);
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];

	  /* Fixed Interface Charge */
	  pNode->qf = 0.0;

	  /* Narrowing of Energy Band-Gap */
	  if (BandGapNarrowing) {
	    absNetConc = ABS(pNode->netConc);
	    if (pNode->netConc > 0.0) {
	      temp1 = log(absNetConc / info->nrefBGN[ELEC]);
	      deltaEg = -info->dEgDn[ELEC] * (temp1 + sqrt(temp1 * temp1 + 0.5));
	      pNode->eg = info->eg0 + deltaEg;
	    } else if (pNode->netConc < 0.0) {
	      temp1 = log(absNetConc / info->nrefBGN[HOLE]);
	      deltaEg = -info->dEgDn[HOLE] * (temp1 + sqrt(temp1 * temp1 + 0.5));
	      pNode->eg = info->eg0 + deltaEg;
	    } else {
	      pNode->eg = info->eg0;
	    }
	  } else {
	    pNode->eg = info->eg0;
	  }
	  pNode->nie = ncv0 * exp(-0.5 * pNode->eg / Vt);
	  pNode->eaff = info->affin;
	  /* Save band structure parameter. */
	  psiBand[index] = -info->refPsi;

	  /* Ionized-Impurity-Scattering Reduction of Carrier Lifetime */
	  if (ConcDepLifetime) {
	    totalConc = pNode->totalConc;
	    temp1 = 1.0 / (1.0 + totalConc / info->nrefSRH[ELEC]);
	    pNode->tn = info->tau0[ELEC] * temp1;
	    temp1 = 1.0 / (1.0 + totalConc / info->nrefSRH[HOLE]);
	    pNode->tp = info->tau0[HOLE] * temp1;
	  } else {
	    pNode->tn = info->tau0[ELEC];
	    pNode->tp = info->tau0[HOLE];
	  }
	}
      }

      pEdge = pElem->pEdge;

      /* Variable Band Built-In Potential */
      dBand = psiBand[1] - psiBand[0];
      dNie = log(pElem->pNodes[1]->nie / pElem->pNodes[0]->nie);
      pEdge->dCBand = dBand + dNie;
      pEdge->dVBand = -dBand + dNie;

      /* Evaluate conc.-dep. mobility. */
      avgConc = 0.5 * (pElem->pRightNode->totalConc + pElem->pLeftNode->totalConc);
      MOBconcDep(info, avgConc, &pEdge->mun, &pEdge->mup);
    }
  }
}

/* Transfer BC info from bdry to nodes and edges. */
static void
ONEcopyBCinfo(ONEdevice *pDevice, ONEelem *pElem, BDRYcard *bdry, int index)
{
  ONEnode *pNode;
  ONEelem *pNElem;
  int eIndex;
  double length;

  NG_IGNORE(pDevice);

  /* First add fixed charge. */
  pNode = pElem->pNodes[index];
  pNode->qf += bdry->BDRYqf;

  /* Now add surface recombination. */
  /* Compute semiconductor length around this node. */
  length = 0.0;
  for (eIndex = 0; eIndex <= 1; eIndex++) {
    pNElem = pNode->pElems[eIndex];
    if ((pNElem != NULL) && (pElem->elemType == SEMICON)) {
      length += 0.5 * pElem->dx;
    }
  }
  if (bdry->BDRYsnGiven) {
    pNode->tn = pNode->tn /
	(1.0 + ((bdry->BDRYsn * TNorm) * pNode->tn) / length);
  }
  if (bdry->BDRYspGiven) {
    pNode->tp = pNode->tp /
	(1.0 + ((bdry->BDRYsp * TNorm) * pNode->tp) / length);
  }
  /* Finally, surface layer is irrelevant for 1d devices. */
}


/* Compute boundary condition parameters. */
void
ONEsetBCparams(ONEdevice *pDevice, BDRYcard *bdryList, CONTcard *contList)
{
  int index, xIndex;
  ONEelem *pElem, *pNElem;
  BDRYcard *bdry;
  CONTcard *cont;

  for (bdry = bdryList; bdry != NULL; bdry = bdry->BDRYnextCard) {
    for (xIndex = bdry->BDRYixLow; xIndex < bdry->BDRYixHigh; xIndex++) {
      pElem = pDevice->elemArray[xIndex];
      if (pElem != NULL) {
	if (pElem->domain == bdry->BDRYdomain) {
	  for (index = 0; index <= 1; index++) {
	    if (pElem->evalNodes[index]) {
	      pNElem = pElem->pElems[index];
	      if (bdry->BDRYneighborGiven) {
		if (pNElem && (pNElem->domain == bdry->BDRYneighbor)) {
		  /* Found an interface node. */
		  ONEcopyBCinfo(pDevice, pElem, bdry, index);
		}
	      } else {
		if ((!pNElem) || (pNElem->domain != pElem->domain)) {
		  /* Found a boundary node. */
		  ONEcopyBCinfo(pDevice, pElem, bdry, index);
		}
	      }
	    }
	  }
	}
      }
    }
  }
  for (cont = contList; cont != NULL; cont = cont->CONTnextCard) {
    if (!cont->CONTworkfunGiven) {
      cont->CONTworkfun = PHI_METAL;
    }
    /*
     * XXX This won't work right if someone tries to change the 1d BJT base
     * contact workfunction and doesn't want to change the emitter. But no
     * one will probably try to do that.
     */
    if (cont->CONTnumber == 1) {
      pDevice->elemArray[1]->pNodes[0]->eaff = cont->CONTworkfun;
    } else if ((cont->CONTnumber == 2) || (cont->CONTnumber == 3)) {
      pDevice->elemArray[pDevice->numNodes - 1]->pNodes[1]->eaff =
	  cont->CONTworkfun;
    }
  }
}

void
ONEnormalize(ONEdevice *pDevice)
{
  int index, eIndex;
  ONEelem *pElem;
  ONEnode *pNode;

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];

    pElem->dx /= LNorm;
    pElem->rDx = 1.0 / pElem->dx;
    pElem->epsRel /= EpsNorm;
    for (index = 0; index <= 1; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	pNode->netConc /= NNorm;
	pNode->nd /= NNorm;
	pNode->na /= NNorm;
	pNode->qf /= (NNorm * LNorm);
	pNode->nie /= NNorm;
	pNode->eg /= VNorm;
	pNode->eaff /= VNorm;
      }
    }
  }
}
