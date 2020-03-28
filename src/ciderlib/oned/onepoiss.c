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

#ifdef KLU
#include "ngspice/klu-binding.h"
#endif

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

#ifdef KLU
    pNode->fPsiPsi = SMPmakeEltKLUforCIDER (matrix, pNode->poiEqn, pNode->poiEqn) ;
    pNode->fPsiPsiBinding = NULL ;
#else
    pNode->fPsiPsi = SMPmakeElt(matrix, pNode->poiEqn, pNode->poiEqn);
#endif

    pNode1 = pElem->pRightNode;

#ifdef KLU
    pNode->fPsiPsiiP1 = SMPmakeEltKLUforCIDER (matrix, pNode->poiEqn, pNode1->poiEqn) ;
    pNode->fPsiPsiiP1Binding = NULL ;
#else
    pNode->fPsiPsiiP1 = SMPmakeElt(matrix, pNode->poiEqn, pNode1->poiEqn);
#endif

    pNode = pElem->pRightNode;

#ifdef KLU
    pNode->fPsiPsi = SMPmakeEltKLUforCIDER (matrix, pNode->poiEqn, pNode->poiEqn) ;
    pNode->fPsiPsiBinding = NULL ;
#else
    pNode->fPsiPsi = SMPmakeElt(matrix, pNode->poiEqn, pNode->poiEqn);
#endif

    pNode1 = pElem->pLeftNode;

#ifdef KLU
    pNode->fPsiPsiiM1 = SMPmakeEltKLUforCIDER (matrix, pNode->poiEqn, pNode1->poiEqn) ;
    pNode->fPsiPsiiM1Binding = NULL ;
#else
    pNode->fPsiPsiiM1 = SMPmakeElt(matrix, pNode->poiEqn, pNode1->poiEqn);
#endif

  }
}

#ifdef KLU
/*
#define CREATE_KLU_BINDING_TABLE_CIDER(ptr, binding, a, b)                             \
    printf ("Swapping Pointer %s: (%d,%d)\n", #ptr, a, b) ; \
    if ((a > 0) && (b > 0)) {                                      \
        if (pNode->binding != NULL) { \
            if (pNode->binding->CSC_Complex != NULL) { \
                printf ("  Looking for the Pointer: %p\n", pNode->binding->CSC_Complex) ; \
                qsort (BindStructCSC, nz, sizeof(BindKluElementCOO), BindKluCompareCSC) ; \
                i.COO = NULL ; \
                i.CSC_Complex = pNode->binding->CSC_Complex ; \
                matched = (BindKluElementCOO *) bsearch (&i, BindStructCSC, nz, sizeof(BindKluElementCOO), BindKluCompareCSC) ; \
                if (matched != NULL) { \
                    printf ("  Found the Old Pointer\n") ; \
                    pNode->ptr = pNode->binding->CSC_Complex ; \
                } else { \
                    i.COO = pNode->ptr ;                                                          \
                    i.CSC_Complex = NULL ; \
                    matched = (BindKluElementCOO *) bsearch (&i, BindStruct, nz, sizeof(BindKluElementCOO), BindKluCompareCOO) ; \
                    if (matched != NULL) { \
                        printf ("  Looking for the Pointer 1\n") ; \
                        pNode->binding = matched ;                                                \
                        pNode->ptr = matched->CSC_Complex ;                                               \
                    } else { \
                        printf ("  Leaving the Pointer as is\n") ; \
                    } \
                } \
            } else { \
                printf ("  Looking for the Pointer 2\n") ; \
                i.COO = pNode->ptr ;                                                          \
                i.CSC_Complex = NULL ; \
                matched = (BindKluElementCOO *) bsearch (&i, BindStruct, nz, sizeof(BindKluElementCOO), BindKluCompareCOO) ; \
                pNode->binding = matched ;                                                \
                pNode->ptr = matched->CSC_Complex ;                                               \
            } \
        } else { \
            printf ("  Looking for the Pointer 3\n") ; \
            i.COO = pNode->ptr ;                                                          \
            i.CSC_Complex = NULL ; \
            matched = (BindKluElementCOO *) bsearch (&i, BindStruct, nz, sizeof(BindKluElementCOO), BindKluCompareCOO) ; \
            pNode->binding = matched ;                                                \
            pNode->ptr = matched->CSC_Complex ;                                               \
        } \
    }
*/

void
ONEQbindCSC (ONEdevice *pDevice)
{
  ONEelem *pElem ;
  ONEnode *pNode, *pNode1 ;
  int index ;
  BindKluElementCOO i, *matched, *BindStruct, *BindStructCSC ;
  size_t nz ;

  BindStruct = pDevice->matrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
  nz = pDevice->matrix->SMPkluMatrix->KLUmatrixNZ ;

  BindStructCSC = (BindKluElementCOO *) malloc (nz * sizeof(BindKluElementCOO)) ;
  for (index = 0 ; index < (int)nz ; index++) {
    BindStructCSC [index] = BindStruct [index] ;
  }

  for (index = 1 ; index < pDevice->numNodes ; index++) {
    pElem = pDevice->elemArray [index] ;
    pNode = pElem->pLeftNode ;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsi, fPsiPsiBinding, pNode->poiEqn, pNode->poiEqn) ;

    pNode1 = pElem->pRightNode ;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiP1, fPsiPsiiP1Binding, pNode->poiEqn, pNode1->poiEqn) ;

    pNode = pElem->pRightNode ;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsi, fPsiPsiBinding, pNode->poiEqn, pNode->poiEqn) ;

    pNode1 = pElem->pLeftNode ;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiM1, fPsiPsiiM1Binding, pNode->poiEqn, pNode1->poiEqn) ;
  }

  free (BindStructCSC) ;
}
#endif

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
#ifdef KLU
  if (pDevice->matrix->CKTkluMODE) {
    SMPclearKLUforCIDER (pDevice->matrix) ;
  } else {
#endif

    SMPclear(pDevice->matrix);

#ifdef KLU
  }
#endif

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
