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

#ifdef KLU
#include "ngspice/klu-binding.h"
#endif


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

#ifdef KLU
      pNode->fPsiPsi = SMPmakeEltKLUforCIDER (matrix, psiEqn, psiEqn) ;
      pNode->fPsiPsiBinding = NULL ;
#else
      pNode->fPsiPsi = SMPmakeElt(matrix, psiEqn, psiEqn);
#endif

      if (pElem->elemType == SEMICON) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pEqn = pNode->pEqn;
	/* pointers for additional terms */

#ifdef KLU
        pNode->fPsiN = SMPmakeEltKLUforCIDER (matrix, psiEqn, nEqn) ;
        pNode->fPsiNBinding = NULL ;
#else
	pNode->fPsiN = SMPmakeElt(matrix, psiEqn, nEqn);
#endif

#ifdef KLU
        pNode->fPsiP = SMPmakeEltKLUforCIDER (matrix, psiEqn, pEqn) ;
        pNode->fPsiPBinding = NULL ;
#else
	pNode->fPsiP = SMPmakeElt(matrix, psiEqn, pEqn);
#endif

#ifdef KLU
        pNode->fNPsi = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqn) ;
        pNode->fNPsiBinding = NULL ;
#else
	pNode->fNPsi = SMPmakeElt(matrix, nEqn, psiEqn);
#endif

#ifdef KLU
        pNode->fNN = SMPmakeEltKLUforCIDER (matrix, nEqn, nEqn) ;
        pNode->fNNBinding = NULL ;
#else
	pNode->fNN = SMPmakeElt(matrix, nEqn, nEqn);
#endif

#ifdef KLU
        pNode->fNP = SMPmakeEltKLUforCIDER (matrix, nEqn, pEqn) ;
        pNode->fNPBinding = NULL ;
#else
	pNode->fNP = SMPmakeElt(matrix, nEqn, pEqn);
#endif

#ifdef KLU
        pNode->fPPsi = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqn) ;
        pNode->fPPsiBinding = NULL ;
#else
	pNode->fPPsi = SMPmakeElt(matrix, pEqn, psiEqn);
#endif

#ifdef KLU
        pNode->fPP = SMPmakeEltKLUforCIDER (matrix, pEqn, pEqn) ;
        pNode->fPPBinding = NULL ;
#else
	pNode->fPP = SMPmakeElt(matrix, pEqn, pEqn);
#endif

#ifdef KLU
        pNode->fPN = SMPmakeEltKLUforCIDER (matrix, pEqn, nEqn) ;
        pNode->fPNBinding = NULL ;
#else
	pNode->fPN = SMPmakeElt(matrix, pEqn, nEqn);
#endif

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

#ifdef KLU
    pNode->fPsiPsiiP1 = SMPmakeEltKLUforCIDER (matrix, psiEqnL, psiEqnR) ;
    pNode->fPsiPsiiP1Binding = NULL ;
#else
    pNode->fPsiPsiiP1 = SMPmakeElt(matrix, psiEqnL, psiEqnR);
#endif

    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */

#ifdef KLU
      pNode->fNPsiiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnL, psiEqnR) ;
      pNode->fNPsiiP1Binding = NULL ;
#else
      pNode->fNPsiiP1 = SMPmakeElt(matrix, nEqnL, psiEqnR);
#endif

#ifdef KLU
      pNode->fNNiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnL, nEqnR) ;
      pNode->fNNiP1Binding = NULL ;
#else
      pNode->fNNiP1 = SMPmakeElt(matrix, nEqnL, nEqnR);
#endif

#ifdef KLU
      pNode->fPPsiiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnL, psiEqnR) ;
      pNode->fPPsiiP1Binding = NULL ;
#else
      pNode->fPPsiiP1 = SMPmakeElt(matrix, pEqnL, psiEqnR);
#endif

#ifdef KLU
      pNode->fPPiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnL, pEqnR) ;
      pNode->fPPiP1Binding = NULL ;
#else
      pNode->fPPiP1 = SMPmakeElt(matrix, pEqnL, pEqnR);
#endif

      if (AvalancheGen) {

#ifdef KLU
        pNode->fNPiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnL, pEqnR) ;
        pNode->fNPiP1Binding = NULL ;
#else
	pNode->fNPiP1 = SMPmakeElt(matrix, nEqnL, pEqnR);
#endif

#ifdef KLU
        pNode->fPNiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnL, nEqnR) ;
        pNode->fPNiP1Binding = NULL ;
#else
	pNode->fPNiP1 = SMPmakeElt(matrix, pEqnL, nEqnR);
#endif

      }
    }
    pNode = pElem->pRightNode;

#ifdef KLU
    pNode->fPsiPsiiM1 = SMPmakeEltKLUforCIDER (matrix, psiEqnR, psiEqnL) ;
    pNode->fPsiPsiiM1Binding = NULL ;
#else
    pNode->fPsiPsiiM1 = SMPmakeElt(matrix, psiEqnR, psiEqnL);
#endif

    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */

#ifdef KLU
      pNode->fNPsiiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnR, psiEqnL) ;
      pNode->fNPsiiM1Binding = NULL ;
#else
      pNode->fNPsiiM1 = SMPmakeElt(matrix, nEqnR, psiEqnL);
#endif

#ifdef KLU
      pNode->fNNiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnR, nEqnL) ;
      pNode->fNNiM1Binding = NULL ;
#else
      pNode->fNNiM1 = SMPmakeElt(matrix, nEqnR, nEqnL);
#endif

#ifdef KLU
      pNode->fPPsiiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnR, psiEqnL) ;
      pNode->fPPsiiM1Binding = NULL ;
#else
      pNode->fPPsiiM1 = SMPmakeElt(matrix, pEqnR, psiEqnL);
#endif

#ifdef KLU
      pNode->fPPiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnR, pEqnL) ;
      pNode->fPPiM1Binding = NULL ;
#else
      pNode->fPPiM1 = SMPmakeElt(matrix, pEqnR, pEqnL);
#endif

      if (AvalancheGen) {

#ifdef KLU
        pNode->fNPiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnR, pEqnL) ;
        pNode->fNPiM1Binding = NULL ;
#else
	pNode->fNPiM1 = SMPmakeElt(matrix, nEqnR, pEqnL);
#endif

#ifdef KLU
        pNode->fPNiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnR, nEqnL) ;
        pNode->fPNiM1Binding = NULL ;
#else
	pNode->fPNiM1 = SMPmakeElt(matrix, pEqnR, nEqnL);
#endif

      }
    }
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

/*
#define CREATE_KLU_BINDING_TABLE_CIDER_TO_REAL(ptr, binding, a, b)                             \
    if ((a > 0) && (b > 0)) {                                      \
        printf ("Macro\n") ; \
        if (pNode->binding) { \
            printf ("IF: %p\n", pNode->binding) ; \
            printf ("COO: %p\n", pNode->binding->COO) ; \
            printf ("CSC: %p\n", pNode->binding->CSC) ; \
            if (pNode->binding->CSC_Complex) { \
                printf ("CSC_Complex: %p\n", pNode->binding->CSC_Complex) ; \
                pNode->ptr = pNode->binding->CSC_Complex ; \
            } else { \
                i = pNode->ptr ;                                                          \
                matched = (BindKluElementCOO *) bsearch (&i, BindStruct, nz, sizeof(BindKluElementCOO), BindKluCompareCOO) ; \
                pNode->binding = matched ;                                                \
                pNode->ptr = matched->CSC_Complex ;                                               \
            } \
        } else { \
            i = pNode->ptr ;                                                          \
            matched = (BindKluElementCOO *) bsearch (&i, BindStruct, nz, sizeof(BindKluElementCOO), BindKluCompareCOO) ; \
            pNode->binding = matched ;                                                \
            pNode->ptr = matched->CSC_Complex ;                                               \
        } \
    }
*/

void
ONEbindCSC (ONEdevice *pDevice)
{
  ONEelem *pElem;
  ONEnode *pNode;
  int index, eIndex;
  int psiEqn, nEqn, pEqn;	/* scratch for deref'd eqn numbers */
  int psiEqnL=0, nEqnL=0, pEqnL=0;
  int psiEqnR=0, nEqnR=0, pEqnR=0;
  BindKluElementCOO i, *matched, *BindStruct, *BindStructCSC ;
  size_t nz ;

  BindStruct = pDevice->matrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
  nz = pDevice->matrix->SMPkluMatrix->KLUmatrixNZ ;

  BindStructCSC = (BindKluElementCOO *) malloc (nz * sizeof(BindKluElementCOO)) ;
  for (index = 0 ; index < (int)nz ; index++) {
    BindStructCSC [index] = BindStruct [index] ;
  }

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    /* first the self terms */
    for (index = 0; index <= 1; index++) {
      pNode = pElem->pNodes[index];
      /* get poisson pointer */
      psiEqn = pNode->psiEqn;

      CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsi, fPsiPsiBinding, psiEqn, psiEqn) ;

      if (pElem->elemType == SEMICON) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pEqn = pNode->pEqn;
	/* pointers for additional terms */

        CREATE_KLU_BINDING_TABLE_CIDER(fPsiN, fPsiNBinding, psiEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPsiP, fPsiPBinding, psiEqn, pEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNPsi, fNPsiBinding, nEqn, psiEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNN, fNNBinding, nEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNP, fNPBinding, nEqn, pEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsi, fPPsiBinding, pEqn, psiEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPP, fPPBinding, pEqn, pEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPN, fPNBinding, pEqn, nEqn) ;

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

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiP1, fPsiPsiiP1Binding, psiEqnL, psiEqnR) ;

    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1, fNPsiiP1Binding, nEqnL, psiEqnR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1, fNNiP1Binding, nEqnL, nEqnR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiP1, fPPsiiP1Binding, pEqnL, psiEqnR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiP1, fPPiP1Binding, pEqnL, pEqnR) ;

      if (AvalancheGen) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPiP1, fNPiP1Binding, nEqnL, pEqnR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPNiP1, fPNiP1Binding, pEqnL, nEqnR) ;

      }
    }
    pNode = pElem->pRightNode;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiM1, fPsiPsiiM1Binding, psiEqnR, psiEqnL) ;

    if (pElem->elemType == SEMICON) {
      /* pointers for additional terms */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1, fNPsiiM1Binding, nEqnR, psiEqnL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1, fNNiM1Binding, nEqnR, nEqnL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiM1, fPPsiiM1Binding, pEqnR, psiEqnL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiM1, fPPiM1Binding, pEqnR, pEqnL) ;

      if (AvalancheGen) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPiM1, fNPiM1Binding, nEqnR, pEqnL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPNiM1, fPNiM1Binding, pEqnR, nEqnL) ;

      }
    }
  }

  free (BindStructCSC) ;
}
#endif

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
#ifdef KLU
  if (pDevice->matrix->CKTkluMODE) {
    SMPclearKLUforCIDER (pDevice->matrix) ;
  } else {
#endif

    SMPclear(pDevice->matrix);

#ifdef KLU
  }
#endif

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
#ifdef KLU
  if (pDevice->matrix->CKTkluMODE) {
    SMPclearKLUforCIDER (pDevice->matrix) ;
  } else {
#endif

    SMPclear(pDevice->matrix);

#ifdef KLU
  }
#endif

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
