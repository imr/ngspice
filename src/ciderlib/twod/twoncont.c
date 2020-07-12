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
#include "ngspice/bool.h"
#include "ngspice/spmatrix.h"
#include "twoddefs.h"
#include "twodext.h"
#include "ngspice/cidersupt.h"
#include "../../maths/misc/bernoull.h"

#ifdef KLU
#include "ngspice/klu-binding.h"
#endif

/*
 * Functions to setup and solve the continuity equations.
 * Both continuity equations are solved.
 * Separate functions are used for one continuity equation.
 */


/*
 * Setup matrix pointers to Jacobian entries and 
 * store direct pointers with the nodes.
 */

void 
  TWONjacBuild(TWOdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOchannel *pCh;
  int eIndex, nIndex;
  int nextIndex;			/* index of node to find next element */
  int psiEqn, nEqn;			/* scratch for deref'd eqn numbers */
  int psiEqnTL = 0, nEqnTL = 0;
  int psiEqnTR = 0, nEqnTR = 0;
  int psiEqnBR = 0, nEqnBR = 0;
  int psiEqnBL = 0, nEqnBL = 0;
  int psiEqnInM = 0, psiEqnInP = 0;		/* scratch for deref'd surface eqns */
  int psiEqnOxM = 0, psiEqnOxP = 0;		/* M= more negative, P= more positive */
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    
    /* first the self terms */
    for ( nIndex = 0; nIndex <= 3; nIndex++ ) {
      pNode = pElem->pNodes[ nIndex ];
      /* get poisson-only pointer */
      psiEqn = pNode->psiEqn;

#ifdef KLU
      pNode->fPsiPsi = SMPmakeEltKLUforCIDER (matrix, psiEqn, psiEqn) ;
      pNode->fPsiPsiBinding = NULL ;
#else
      pNode->fPsiPsi = SMPmakeElt(matrix, psiEqn, psiEqn);
#endif

      if ( pElem->elemType == SEMICON ) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pNode->pEqn = 0;		/* Throw pEqn number into garbage. */
	/* pointers for additional terms */

#ifdef KLU
        pNode->fPsiN = SMPmakeEltKLUforCIDER (matrix, psiEqn, nEqn) ;
        pNode->fPsiNBinding = NULL ;
#else
	pNode->fPsiN = SMPmakeElt(matrix, psiEqn, nEqn);
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

      } else {
	nEqn = 0;
      }
      /* save equation indices */
      switch ( nIndex ) {
      case 0: /* TL Node */
	psiEqnTL = psiEqn;
	nEqnTL = nEqn;
	break;
      case 1: /* TR Node */
	psiEqnTR = psiEqn;
	nEqnTR = nEqn;
	break;
      case 2: /* BR Node */
	psiEqnBR = psiEqn;
	nEqnBR = nEqn;
	break;
      case 3: /* BL Node */
	psiEqnBL = psiEqn;
	nEqnBL = nEqn;
	break;
      default:
	break;
      }
    }
    
    /* now terms to couple to adjacent nodes */
    pNode = pElem->pTLNode;

#ifdef KLU
    pNode->fPsiPsiiP1 = SMPmakeEltKLUforCIDER (matrix, psiEqnTL, psiEqnTR) ;
    pNode->fPsiPsiiP1Binding = NULL ;
#else
    pNode->fPsiPsiiP1 = SMPmakeElt(matrix, psiEqnTL, psiEqnTR);
#endif

#ifdef KLU
    pNode->fPsiPsijP1 = SMPmakeEltKLUforCIDER (matrix, psiEqnTL, psiEqnBL) ;
    pNode->fPsiPsijP1Binding = NULL ;
#else
    pNode->fPsiPsijP1 = SMPmakeElt(matrix, psiEqnTL, psiEqnBL);
#endif

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

#ifdef KLU
      pNode->fNPsiiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, psiEqnTR) ;
      pNode->fNPsiiP1Binding = NULL ;
#else
      pNode->fNPsiiP1 = SMPmakeElt(matrix, nEqnTL, psiEqnTR);
#endif

#ifdef KLU
      pNode->fNNiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, nEqnTR) ;
      pNode->fNNiP1Binding = NULL ;
#else
      pNode->fNNiP1 = SMPmakeElt(matrix, nEqnTL, nEqnTR);
#endif

#ifdef KLU
      pNode->fNPsijP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, psiEqnBL) ;
      pNode->fNPsijP1Binding = NULL ;
#else
      pNode->fNPsijP1 = SMPmakeElt(matrix, nEqnTL, psiEqnBL);
#endif

#ifdef KLU
      pNode->fNNjP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, nEqnBL) ;
      pNode->fNNjP1Binding = NULL ;
#else
      pNode->fNNjP1 = SMPmakeElt(matrix, nEqnTL, nEqnBL);
#endif

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

#ifdef KLU
        pNode->fNPsiiP1jP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, psiEqnBR) ;
        pNode->fNPsiiP1jP1Binding = NULL ;
#else
        pNode->fNPsiiP1jP1 = SMPmakeElt(matrix, nEqnTL, psiEqnBR);
#endif

#ifdef KLU
        pNode->fNNiP1jP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTL, nEqnBR) ;
        pNode->fNNiP1jP1Binding = NULL ;
#else
        pNode->fNNiP1jP1 = SMPmakeElt(matrix, nEqnTL, nEqnBR);
#endif

      }
    }
    
    pNode = pElem->pTRNode;

#ifdef KLU
    pNode->fPsiPsiiM1 = SMPmakeEltKLUforCIDER (matrix, psiEqnTR, psiEqnTL) ;
    pNode->fPsiPsiiM1Binding = NULL ;
#else
    pNode->fPsiPsiiM1 = SMPmakeElt(matrix, psiEqnTR, psiEqnTL);
#endif

#ifdef KLU
    pNode->fPsiPsijP1 = SMPmakeEltKLUforCIDER (matrix, psiEqnTR, psiEqnBR) ;
    pNode->fPsiPsijP1Binding = NULL ;
#else
    pNode->fPsiPsijP1 = SMPmakeElt(matrix, psiEqnTR, psiEqnBR);
#endif

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

#ifdef KLU
      pNode->fNPsiiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, psiEqnTL) ;
      pNode->fNPsiiM1Binding = NULL ;
#else
      pNode->fNPsiiM1 = SMPmakeElt(matrix, nEqnTR, psiEqnTL);
#endif

#ifdef KLU
      pNode->fNNiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, nEqnTL) ;
      pNode->fNNiM1Binding = NULL ;
#else
      pNode->fNNiM1 = SMPmakeElt(matrix, nEqnTR, nEqnTL);
#endif

#ifdef KLU
      pNode->fNPsijP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, psiEqnBR) ;
      pNode->fNPsijP1Binding = NULL ;
#else
      pNode->fNPsijP1 = SMPmakeElt(matrix, nEqnTR, psiEqnBR);
#endif

#ifdef KLU
      pNode->fNNjP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, nEqnBR) ;
      pNode->fNNjP1Binding = NULL ;
#else
      pNode->fNNjP1 = SMPmakeElt(matrix, nEqnTR, nEqnBR);
#endif

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

#ifdef KLU
        pNode->fNPsiiM1jP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, psiEqnBL) ;
        pNode->fNPsiiM1jP1Binding = NULL ;
#else
        pNode->fNPsiiM1jP1 = SMPmakeElt(matrix, nEqnTR, psiEqnBL);
#endif

#ifdef KLU
        pNode->fNNiM1jP1 = SMPmakeEltKLUforCIDER (matrix, nEqnTR, nEqnBL) ;
        pNode->fNNiM1jP1Binding = NULL ;
#else
        pNode->fNNiM1jP1 = SMPmakeElt(matrix, nEqnTR, nEqnBL);
#endif

      }
    }
    
    pNode = pElem->pBRNode;

#ifdef KLU
    pNode->fPsiPsiiM1 = SMPmakeEltKLUforCIDER (matrix, psiEqnBR, psiEqnBL) ;
    pNode->fPsiPsiiM1Binding = NULL ;
#else
    pNode->fPsiPsiiM1 = SMPmakeElt(matrix, psiEqnBR, psiEqnBL);
#endif

#ifdef KLU
    pNode->fPsiPsijM1 = SMPmakeEltKLUforCIDER (matrix, psiEqnBR, psiEqnTR) ;
    pNode->fPsiPsijM1Binding = NULL ;
#else
    pNode->fPsiPsijM1 = SMPmakeElt(matrix, psiEqnBR, psiEqnTR);
#endif

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

#ifdef KLU
      pNode->fNPsiiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, psiEqnBL) ;
      pNode->fNPsiiM1Binding = NULL ;
#else
      pNode->fNPsiiM1 = SMPmakeElt(matrix, nEqnBR, psiEqnBL);
#endif

#ifdef KLU
      pNode->fNNiM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, nEqnBL) ;
      pNode->fNNiM1Binding = NULL ;
#else
      pNode->fNNiM1 = SMPmakeElt(matrix, nEqnBR, nEqnBL);
#endif

#ifdef KLU
      pNode->fNPsijM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, psiEqnTR) ;
      pNode->fNPsijM1Binding = NULL ;
#else
      pNode->fNPsijM1 = SMPmakeElt(matrix, nEqnBR, psiEqnTR);
#endif

#ifdef KLU
      pNode->fNNjM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, nEqnTR) ;
      pNode->fNNjM1Binding = NULL ;
#else
      pNode->fNNjM1 = SMPmakeElt(matrix, nEqnBR, nEqnTR);
#endif

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

#ifdef KLU
        pNode->fNPsiiM1jM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, psiEqnTL) ;
        pNode->fNPsiiM1jM1Binding = NULL ;
#else
        pNode->fNPsiiM1jM1 = SMPmakeElt(matrix, nEqnBR, psiEqnTL);
#endif

#ifdef KLU
        pNode->fNNiM1jM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBR, nEqnTL) ;
        pNode->fNNiM1jM1Binding = NULL ;
#else
        pNode->fNNiM1jM1 = SMPmakeElt(matrix, nEqnBR, nEqnTL);
#endif

      }
    }
    
    pNode = pElem->pBLNode;

#ifdef KLU
    pNode->fPsiPsiiP1 = SMPmakeEltKLUforCIDER (matrix, psiEqnBL, psiEqnBR) ;
    pNode->fPsiPsiiP1Binding = NULL ;
#else
    pNode->fPsiPsiiP1 = SMPmakeElt(matrix, psiEqnBL, psiEqnBR);
#endif

#ifdef KLU
    pNode->fPsiPsijM1 = SMPmakeEltKLUforCIDER (matrix, psiEqnBL, psiEqnTL) ;
    pNode->fPsiPsijM1Binding = NULL ;
#else
    pNode->fPsiPsijM1 = SMPmakeElt(matrix, psiEqnBL, psiEqnTL);
#endif

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

#ifdef KLU
      pNode->fNPsiiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, psiEqnBR) ;
      pNode->fNPsiiP1Binding = NULL ;
#else
      pNode->fNPsiiP1 = SMPmakeElt(matrix, nEqnBL, psiEqnBR);
#endif

#ifdef KLU
      pNode->fNNiP1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, nEqnBR) ;
      pNode->fNNiP1Binding = NULL ;
#else
      pNode->fNNiP1 = SMPmakeElt(matrix, nEqnBL, nEqnBR);
#endif

#ifdef KLU
      pNode->fNPsijM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, psiEqnTL) ;
      pNode->fNPsijM1Binding = NULL ;
#else
      pNode->fNPsijM1 = SMPmakeElt(matrix, nEqnBL, psiEqnTL);
#endif

#ifdef KLU
      pNode->fNNjM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, nEqnTL) ;
      pNode->fNNjM1Binding = NULL ;
#else
      pNode->fNNjM1 = SMPmakeElt(matrix, nEqnBL, nEqnTL);
#endif

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

#ifdef KLU
        pNode->fNPsiiP1jM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, psiEqnTR) ;
        pNode->fNPsiiP1jM1Binding = NULL ;
#else
        pNode->fNPsiiP1jM1 = SMPmakeElt(matrix, nEqnBL, psiEqnTR);
#endif

#ifdef KLU
        pNode->fNNiP1jM1 = SMPmakeEltKLUforCIDER (matrix, nEqnBL, nEqnTR) ;
        pNode->fNNiP1jM1Binding = NULL ;
#else
        pNode->fNNiP1jM1 = SMPmakeElt(matrix, nEqnBL, nEqnTR);
#endif

      }
    }
  }
  /* 
   * Add terms for surface-field of inversion-layer mobility model.
   * Elements MUST be made from silicon for this to work.
   * No empty elements are allowed.
   * Don't need these pointers if SurfaceMobility isn't set.
   */
  if ( MobDeriv && SurfaceMobility ) {
    for ( pCh = pDevice->pChannel; pCh != NULL;
	 pCh = pCh->next ) {
      pElem = pCh->pNElem;
      switch (pCh->type) {
      case 0:
	psiEqnInM = pElem->pBLNode->psiEqn;
	psiEqnInP = pElem->pBRNode->psiEqn;
	psiEqnOxM = pElem->pTLNode->psiEqn;
	psiEqnOxP = pElem->pTRNode->psiEqn;
	break;
      case 1:
	psiEqnInM = pElem->pTLNode->psiEqn;
	psiEqnInP = pElem->pBLNode->psiEqn;
	psiEqnOxM = pElem->pTRNode->psiEqn;
	psiEqnOxP = pElem->pBRNode->psiEqn;
	break;
      case 2:
	psiEqnInM = pElem->pTLNode->psiEqn;
	psiEqnInP = pElem->pTRNode->psiEqn;
	psiEqnOxM = pElem->pBLNode->psiEqn;
	psiEqnOxP = pElem->pBRNode->psiEqn;
	break;
      case 3:
	psiEqnInM = pElem->pTRNode->psiEqn;
	psiEqnInP = pElem->pBRNode->psiEqn;
	psiEqnOxM = pElem->pTLNode->psiEqn;
	psiEqnOxP = pElem->pBLNode->psiEqn;
	break;
      }
      pElem = pCh->pSeed;
      nextIndex = (pCh->type + 2)%4;
      while (pElem && pElem->channel == pCh->id) {
	for ( nIndex = 0; nIndex <= 3; nIndex++ ) {
	  pNode = pElem->pNodes[ nIndex ];
	  psiEqn = pNode->psiEqn;
	  nEqn = pNode->nEqn;
	  if ( pCh->type % 2 == 0 ) { /* Vertical Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left Side */

#ifdef KLU
              pNode->fNPsiIn = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInM) ;
              pNode->fNPsiInBinding = NULL ;
#else
              pNode->fNPsiIn = SMPmakeElt(matrix, nEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fNPsiInP1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInP) ;
              pNode->fNPsiInP1Binding = NULL ;
#else
              pNode->fNPsiInP1 = SMPmakeElt(matrix, nEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fNPsiOx = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxM) ;
              pNode->fNPsiOxBinding = NULL ;
#else
              pNode->fNPsiOx = SMPmakeElt(matrix, nEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fNPsiOxP1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxP) ;
              pNode->fNPsiOxP1Binding = NULL ;
#else
              pNode->fNPsiOxP1 = SMPmakeElt(matrix, nEqn, psiEqnOxP);
#endif

	    } else { /* Right Side */

#ifdef KLU
              pNode->fNPsiInM1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInM) ;
              pNode->fNPsiInM1Binding = NULL ;
#else
              pNode->fNPsiInM1 = SMPmakeElt(matrix, nEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fNPsiIn = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInP) ;
              pNode->fNPsiInBinding = NULL ;
#else
              pNode->fNPsiIn = SMPmakeElt(matrix, nEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fNPsiOxM1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxM) ;
              pNode->fNPsiOxM1Binding = NULL ;
#else
              pNode->fNPsiOxM1 = SMPmakeElt(matrix, nEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fNPsiOx = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxP) ;
              pNode->fNPsiOxBinding = NULL ;
#else
              pNode->fNPsiOx = SMPmakeElt(matrix, nEqn, psiEqnOxP);
#endif

	    }
	  } else { /* Horizontal Slice */
	    if ( nIndex <= 1 ) { /* Top Side */

#ifdef KLU
              pNode->fNPsiIn = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInM) ;
              pNode->fNPsiInBinding = NULL ;
#else
              pNode->fNPsiIn = SMPmakeElt(matrix, nEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fNPsiInP1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInP) ;
              pNode->fNPsiInP1Binding = NULL ;
#else
              pNode->fNPsiInP1 = SMPmakeElt(matrix, nEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fNPsiOx = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxM) ;
              pNode->fNPsiOxBinding = NULL ;
#else
              pNode->fNPsiOx = SMPmakeElt(matrix, nEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fNPsiOxP1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxP) ;
              pNode->fNPsiOxP1Binding = NULL ;
#else
              pNode->fNPsiOxP1 = SMPmakeElt(matrix, nEqn, psiEqnOxP);
#endif

	    } else { /* Bottom Side */

#ifdef KLU
              pNode->fNPsiInM1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInM) ;
              pNode->fNPsiInM1Binding = NULL ;
#else
              pNode->fNPsiInM1 = SMPmakeElt(matrix, nEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fNPsiIn = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnInP) ;
              pNode->fNPsiInBinding = NULL ;
#else
              pNode->fNPsiIn = SMPmakeElt(matrix, nEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fNPsiOxM1 = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxM) ;
              pNode->fNPsiOxM1Binding = NULL ;
#else
              pNode->fNPsiOxM1 = SMPmakeElt(matrix, nEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fNPsiOx = SMPmakeEltKLUforCIDER (matrix, nEqn, psiEqnOxP) ;
              pNode->fNPsiOxBinding = NULL ;
#else
              pNode->fNPsiOx = SMPmakeElt(matrix, nEqn, psiEqnOxP);
#endif

	    }
	  }
	} /* endfor nIndex */
	pElem = pElem->pElems[ nextIndex ];
      } /* endwhile pElem */
    } /* endfor pCh */
  } /* endif SurfaceMobility */
}

#ifdef KLU
void
TWONbindCSC (TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOchannel *pCh;
  BindElementKLUforCIDER i, *matched, *BindStruct, *BindStructCSC ;
  int index ;
  size_t nz ;

  int eIndex, nIndex;
  int nextIndex;			/* index of node to find next element */
  int psiEqn, nEqn;			/* scratch for deref'd eqn numbers */
  int psiEqnTL = 0, nEqnTL = 0;
  int psiEqnTR = 0, nEqnTR = 0;
  int psiEqnBR = 0, nEqnBR = 0;
  int psiEqnBL = 0, nEqnBL = 0;
  int psiEqnInM = 0, psiEqnInP = 0;		/* scratch for deref'd surface eqns */
  int psiEqnOxM = 0, psiEqnOxP = 0;		/* M= more negative, P= more positive */

  BindStruct = pDevice->matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER ;
  nz = pDevice->matrix->SMPkluMatrix->KLUmatrixNZ ;

  BindStructCSC = (BindElementKLUforCIDER *) malloc (nz * sizeof (BindElementKLUforCIDER)) ;
  for (index = 0 ; index < (int)nz ; index++) {
    BindStructCSC [index] = BindStruct [index] ;
  }

  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    
    /* first the self terms */
    for ( nIndex = 0; nIndex <= 3; nIndex++ ) {
      pNode = pElem->pNodes[ nIndex ];
      /* get poisson-only pointer */
      psiEqn = pNode->psiEqn;

      CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsi, fPsiPsiBinding, psiEqn, psiEqn) ;

      if ( pElem->elemType == SEMICON ) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pNode->pEqn = 0;		/* Throw pEqn number into garbage. */
	/* pointers for additional terms */

        CREATE_KLU_BINDING_TABLE_CIDER(fPsiN, fPsiNBinding, psiEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNPsi, fNPsiBinding, nEqn, psiEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNN, fNNBinding, nEqn, nEqn) ;

      } else {
	nEqn = 0;
      }
      /* save equation indices */
      switch ( nIndex ) {
      case 0: /* TL Node */
	psiEqnTL = psiEqn;
	nEqnTL = nEqn;
	break;
      case 1: /* TR Node */
	psiEqnTR = psiEqn;
	nEqnTR = nEqn;
	break;
      case 2: /* BR Node */
	psiEqnBR = psiEqn;
	nEqnBR = nEqn;
	break;
      case 3: /* BL Node */
	psiEqnBL = psiEqn;
	nEqnBL = nEqn;
	break;
      default:
	break;
      }
    }
    
    /* now terms to couple to adjacent nodes */
    pNode = pElem->pTLNode;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiP1, fPsiPsiiP1Binding, psiEqnTL, psiEqnTR) ;
    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsijP1, fPsiPsijP1Binding, psiEqnTL, psiEqnBL) ;

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1, fNPsiiP1Binding, nEqnTL, psiEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1, fNNiP1Binding, nEqnTL, nEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNPsijP1, fNPsijP1Binding, nEqnTL, psiEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNjP1, fNNjP1Binding, nEqnTL, nEqnBL) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1jP1, fNPsiiP1jP1Binding, nEqnTL, psiEqnBR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1jP1, fNNiP1jP1Binding, nEqnTL, nEqnBR) ;

      }
    }
    
    pNode = pElem->pTRNode;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiM1, fPsiPsiiM1Binding, psiEqnTR, psiEqnTL) ;
    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsijP1, fPsiPsijP1Binding, psiEqnTR, psiEqnBR) ;

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1, fNPsiiM1Binding, nEqnTR, psiEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1, fNNiM1Binding, nEqnTR, nEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNPsijP1, fNPsijP1Binding, nEqnTR, psiEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNjP1, fNNjP1Binding, nEqnTR, nEqnBR) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1jP1, fNPsiiM1jP1Binding, nEqnTR, psiEqnBL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1jP1, fNNiM1jP1Binding, nEqnTR, nEqnBL) ;

      }
    }
    
    pNode = pElem->pBRNode;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiM1, fPsiPsiiM1Binding, psiEqnBR, psiEqnBL) ;
    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsijM1, fPsiPsijM1Binding, psiEqnBR, psiEqnTR) ;

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1, fNPsiiM1Binding, nEqnBR, psiEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1, fNNiM1Binding, nEqnBR, nEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNPsijM1, fNPsijM1Binding, nEqnBR, psiEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNjM1, fNNjM1Binding, nEqnBR, nEqnTR) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1jM1, fNPsiiM1jM1Binding, nEqnBR, psiEqnTL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1jM1, fNNiM1jM1Binding, nEqnBR, nEqnTL) ;

      }
    }
    
    pNode = pElem->pBLNode;

    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsiiP1, fPsiPsiiP1Binding, psiEqnBL, psiEqnBR) ;
    CREATE_KLU_BINDING_TABLE_CIDER(fPsiPsijM1, fPsiPsijM1Binding, psiEqnBL, psiEqnTL) ;

    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */

      CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1, fNPsiiP1Binding, nEqnBL, psiEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1, fNNiP1Binding, nEqnBL, nEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNPsijM1, fNPsijM1Binding, nEqnBL, psiEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fNNjM1, fNNjM1Binding, nEqnBL, nEqnTL) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1jM1, fNPsiiP1jM1Binding, nEqnBL, psiEqnTR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1jM1, fNNiP1jM1Binding, nEqnBL, nEqnTR) ;

      }
    }
  }
  /* 
   * Add terms for surface-field of inversion-layer mobility model.
   * Elements MUST be made from silicon for this to work.
   * No empty elements are allowed.
   * Don't need these pointers if SurfaceMobility isn't set.
   */
  if ( MobDeriv && SurfaceMobility ) {
    for ( pCh = pDevice->pChannel; pCh != NULL;
	 pCh = pCh->next ) {
      pElem = pCh->pNElem;
      switch (pCh->type) {
      case 0:
	psiEqnInM = pElem->pBLNode->psiEqn;
	psiEqnInP = pElem->pBRNode->psiEqn;
	psiEqnOxM = pElem->pTLNode->psiEqn;
	psiEqnOxP = pElem->pTRNode->psiEqn;
	break;
      case 1:
	psiEqnInM = pElem->pTLNode->psiEqn;
	psiEqnInP = pElem->pBLNode->psiEqn;
	psiEqnOxM = pElem->pTRNode->psiEqn;
	psiEqnOxP = pElem->pBRNode->psiEqn;
	break;
      case 2:
	psiEqnInM = pElem->pTLNode->psiEqn;
	psiEqnInP = pElem->pTRNode->psiEqn;
	psiEqnOxM = pElem->pBLNode->psiEqn;
	psiEqnOxP = pElem->pBRNode->psiEqn;
	break;
      case 3:
	psiEqnInM = pElem->pTRNode->psiEqn;
	psiEqnInP = pElem->pBRNode->psiEqn;
	psiEqnOxM = pElem->pTLNode->psiEqn;
	psiEqnOxP = pElem->pBLNode->psiEqn;
	break;
      }
      pElem = pCh->pSeed;
      nextIndex = (pCh->type + 2)%4;
      while (pElem && pElem->channel == pCh->id) {
	for ( nIndex = 0; nIndex <= 3; nIndex++ ) {
	  pNode = pElem->pNodes[ nIndex ];
	  psiEqn = pNode->psiEqn;
	  nEqn = pNode->nEqn;
	  if ( pCh->type % 2 == 0 ) { /* Vertical Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInP1, fNPsiInP1Binding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxP1, fNPsiOxP1Binding, nEqn, psiEqnOxP) ;

	    } else { /* Right Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInM1, fNPsiInM1Binding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxM1, fNPsiOxM1Binding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxP) ;

	    }
	  } else { /* Horizontal Slice */
	    if ( nIndex <= 1 ) { /* Top Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInP1, fNPsiInP1Binding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxP1, fNPsiOxP1Binding, nEqn, psiEqnOxP) ;

	    } else { /* Bottom Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInM1, fNPsiInM1Binding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxM1, fNPsiOxM1Binding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxP) ;

	    }
	  }
	} /* endfor nIndex */
	pElem = pElem->pElems[ nextIndex ];
      } /* endwhile pElem */
    } /* endfor pCh */
  } /* endif SurfaceMobility */

  free (BindStructCSC) ;
}
#endif

/*
 *  The Jacobian and Rhs are loaded by the following function.
 *  Inputs are the transient analysis flag and the transient
 *  information structure
 */

void 
  TWONsysLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  TWOchannel *pCh;
  int index, eIndex;
  int nextIndex;			/* index of node to find next element */
  double *pRhs = pDevice->rhs;
  double dx, dy, dxdy, dyOverDx, dxOverDy;
  double ds;
  double dPsiT, dPsiB, dPsiL, dPsiR;
  double rhsN;
  double nConc, pConc;
  double perTime = 0.0;
  
  /* first compute the currents and derivatives */
  TWONcommonTerms( pDevice, FALSE, tranAnalysis, info );
  
  /* find reciprocal timestep */
  if ( tranAnalysis ) {
    perTime = info->intCoeff[0];
  }
  
  /* zero the rhs vector */
  for ( index = 1 ; index <= pDevice->numEqns ; index++ ) {
    pRhs[ index ] = 0.0;
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
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    
    dx = 0.5 * pElem->dx;
    dy = 0.5 * pElem->dy;
    dxdy = dx * dy;
    dxOverDy = 0.5 * pElem->epsRel * pElem->dxOverDy;
    dyOverDx = 0.5 * pElem->epsRel * pElem->dyOverDx;
    
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;
    dPsiT = pTEdge->dPsi;
    dPsiB = pBEdge->dPsi;
    dPsiL = pLEdge->dPsi;
    dPsiR = pREdge->dPsi;
    
    /* load for all i,j */
    for ( index = 0; index <= 3; index++ ) {
      pNode = pElem->pNodes[ index ];
      if ( pNode->nodeType != CONTACT ) {
	*(pNode->fPsiPsi) += dyOverDx + dxOverDy;
	if ( index <= 1 ) {
	  pHEdge = pTEdge;
	} else {
	  pHEdge = pBEdge;
	}
	if ( index == 0 || index == 3 ) {
	  pVEdge = pLEdge;
	} else {
	  pVEdge = pREdge;
	}
	/* Add surface state charges. */
	pRhs[ pNode->psiEqn ] += dx * pHEdge->qf;
	pRhs[ pNode->psiEqn ] += dy * pVEdge->qf;
	if ( pElem->elemType == SEMICON ) {
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];

	  *(pNode->fPsiN) += dxdy;
	  *(pNode->fPsiPsi) += dxdy * pConc;
	  *(pNode->fNPsi) -= dy * pHEdge->dJnDpsiP1 + dx * pVEdge->dJnDpsiP1;
	  pRhs[ pNode->psiEqn ] += dxdy * (pNode->netConc + pConc - nConc);
	  
	  /* Handle generation terms */
	  *(pNode->fNN) -= dxdy * pNode->dUdN;
	  *(pNode->fNPsi) += dxdy * pNode->dUdP * pConc;
	  rhsN = - dxdy * pNode->uNet;
	  pRhs[ pNode->nEqn ] -= rhsN;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
	    *(pNode->fNN) -= dxdy * perTime;
	    pRhs[ pNode->nEqn ] += dxdy * pNode->dNdT;
	  }
	}
      }
    }
    
    /* Handle neighbor and edge dependent terms */
    pNode = pElem->pTLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiT - dxOverDy * dPsiL;
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pTEdge->jn + dx * pLEdge->jn;
	*(pNode->fNN) += dy * pTEdge->dJnDn + dx * pLEdge->dJnDn;
	*(pNode->fNPsiiP1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pTEdge->dJnDnP1;
	*(pNode->fNPsijP1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pLEdge->dJnDnP1;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pTEdge->jn + dx * pREdge->jn;
	*(pNode->fNN) += -dy * pTEdge->dJnDnP1 + dx * pREdge->dJnDn;
	*(pNode->fNPsiiM1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pTEdge->dJnDn;
	*(pNode->fNPsijP1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pREdge->dJnDnP1;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pBEdge->jn - dx * pREdge->jn;
	*(pNode->fNN) += -dy * pBEdge->dJnDnP1 - dx * pREdge->dJnDnP1;
	*(pNode->fNPsiiM1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pBEdge->dJnDn;
	*(pNode->fNPsijM1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pREdge->dJnDn;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pBEdge->jn - dx * pLEdge->jn;
	*(pNode->fNN) += dy * pBEdge->dJnDn - dx * pLEdge->dJnDnP1;
	*(pNode->fNPsiiP1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pBEdge->dJnDnP1;
	*(pNode->fNPsijM1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pLEdge->dJnDn;
      }
    }
  }
  
  /* Calculate the Inversion-Layer Mobility Dependent Terms in Jac. */
  if ( MobDeriv && SurfaceMobility ) {
    for ( pCh = pDevice->pChannel; pCh != NULL;
	 pCh = pCh->next ) {
      /* Find effective height of oxide element at interface. */
      if ( pCh->type%2 == 0 ) { /* Vertical slice */
	ds = pCh->pNElem->dy / pCh->pNElem->epsRel;
      } else {			/* Horizontal slice */
	ds = pCh->pNElem->dx / pCh->pNElem->epsRel;
      }
      pElem = pCh->pSeed;
      nextIndex = (pCh->type + 2)%4;
      while (pElem && pElem->channel == pCh->id) {  
	TWONmobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}


/*
 * This function used only for direct method ac analysis.
 * Used to load only the dc Jacobian matrix. Rhs is unaffected
 */

void 
  TWONjacLoad(TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  TWOchannel *pCh;
  int index, eIndex;
  int nextIndex;			/* index of node to find next element */
  double dx, dy, dxdy, dyOverDx, dxOverDy;
  double ds;
  double pConc;
  
  /* first compute the currents and derivatives */
  TWONcommonTerms( pDevice, FALSE, FALSE, NULL );
  
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
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    dx = 0.5 * pElem->dx;
    dy = 0.5 * pElem->dy;
    dxdy = dx * dy;
    dxOverDy = 0.5 * pElem->epsRel * pElem->dxOverDy;
    dyOverDx = 0.5 * pElem->epsRel * pElem->dyOverDx;
    
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;
    
    /* load for all i,j */
    for ( index = 0; index <= 3; index++ ) {
      pNode = pElem->pNodes[ index ];
      if ( pNode->nodeType != CONTACT ) {
	*(pNode->fPsiPsi) += dyOverDx + dxOverDy;
	if ( pElem->elemType == SEMICON ) {
	  if ( index <= 1 ) {
	    pHEdge = pTEdge;
	  } else {
	    pHEdge = pBEdge;
	  }
	  if ( index == 0 || index == 3 ) {
	    pVEdge = pLEdge;
	  } else {
	    pVEdge = pREdge;
	  }
	  pConc = pDevice->devState0 [pNode->nodeP];
	  *(pNode->fPsiN) += dxdy;
	  *(pNode->fPsiPsi) += dxdy * pConc;
	  *(pNode->fNPsi) -= dy * pHEdge->dJnDpsiP1 + dx * pVEdge->dJnDpsiP1;
	  
	  /* Handle generation terms */
	  *(pNode->fNN) -= dxdy * pNode->dUdN;
	  *(pNode->fNPsi) += dxdy * pNode->dUdP * pConc;
	}
      }
    }
    
    /* Handle neighbor and edge dependent terms */
    pNode = pElem->pTLNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += dy * pTEdge->dJnDn + dx * pLEdge->dJnDn;
	*(pNode->fNPsiiP1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pTEdge->dJnDnP1;
	*(pNode->fNPsijP1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pLEdge->dJnDnP1;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += -dy * pTEdge->dJnDnP1 + dx * pREdge->dJnDn;
	*(pNode->fNPsiiM1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pTEdge->dJnDn;
	*(pNode->fNPsijP1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pREdge->dJnDnP1;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += -dy * pBEdge->dJnDnP1 - dx * pREdge->dJnDnP1;
	*(pNode->fNPsiiM1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pBEdge->dJnDn;
	*(pNode->fNPsijM1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pREdge->dJnDn;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += dy * pBEdge->dJnDn - dx * pLEdge->dJnDnP1;
	*(pNode->fNPsiiP1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pBEdge->dJnDnP1;
	*(pNode->fNPsijM1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pLEdge->dJnDn;
      }
    }
  }

  /* Calculate the Inversion-Layer Mobility Dependent Terms in Jac. */
  if ( MobDeriv && SurfaceMobility ) {
    for ( pCh = pDevice->pChannel; pCh != NULL;
	 pCh = pCh->next ) {
      /* Find effective height of oxide element at interface. */
      if ( pCh->type%2 == 0 ) { /* Vertical slice */
	ds = pCh->pNElem->dy / pCh->pNElem->epsRel;
      } else {			/* Horizontal slice */
	ds = pCh->pNElem->dx / pCh->pNElem->epsRel;
      }
      pElem = pCh->pSeed;
      nextIndex = (pCh->type + 2)%4;
      while (pElem && pElem->channel == pCh->id) {  
	TWONmobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}

/* load only the Rhs vector */
void 
  TWONrhsLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dx, dy, dxdy, dyOverDx, dxOverDy;
  double dPsiT, dPsiB, dPsiL, dPsiR;
  double rhsN;
  double nConc, pConc;
  double perTime;
  
  /* first compute the currents */
  TWONcommonTerms( pDevice, TRUE, tranAnalysis, info );
  
  /* find reciprocal timestep */
  if ( tranAnalysis ) {
    perTime = info->intCoeff[0];
  }
  
  /* zero the rhs vector */
  for ( index = 1 ; index <= pDevice->numEqns ; index++ ) {
    pRhs[ index ] = 0.0;
  }
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];

    dx = 0.5 * pElem->dx;
    dy = 0.5 * pElem->dy;
    dxdy = dx * dy;
    dxOverDy = 0.5 * pElem->epsRel * pElem->dxOverDy;
    dyOverDx = 0.5 * pElem->epsRel * pElem->dyOverDx;
    
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;
    dPsiT = pTEdge->dPsi;
    dPsiB = pBEdge->dPsi;
    dPsiL = pLEdge->dPsi;
    dPsiR = pREdge->dPsi;
    
    /* load for all i,j */
    for ( index = 0; index <= 3; index++ ) {
      pNode = pElem->pNodes[ index ];
      if ( pNode->nodeType != CONTACT ) {
	if ( index <= 1 ) {
	  pHEdge = pTEdge;
	} else {
	  pHEdge = pBEdge;
	}
	if ( index == 0 || index == 3 ) {
	  pVEdge = pLEdge;
	} else {
	  pVEdge = pREdge;
	}
	/* Add surface state charges. */
	pRhs[ pNode->psiEqn ] += dx * pHEdge->qf;
	pRhs[ pNode->psiEqn ] += dy * pVEdge->qf;
	if ( pElem->elemType == SEMICON ) {
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];
	  pRhs[ pNode->psiEqn ] += dxdy * (pNode->netConc + pConc - nConc);
	  
	  /* Handle generation terms */
	  rhsN = - dxdy * pNode->uNet;
	  pRhs[ pNode->nEqn ] -= rhsN;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
	    pRhs[ pNode->nEqn ] += dxdy * pNode->dNdT;
	  }
	}
      }
    }
    
    /* Handle neighbor and edge dependent terms */
    pNode = pElem->pTLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiT - dxOverDy * dPsiL;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pTEdge->jn + dx * pLEdge->jn;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pTEdge->jn + dx * pREdge->jn;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pBEdge->jn - dx * pREdge->jn;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pBEdge->jn - dx * pLEdge->jn;
      }
    }
  }
}

/*
 * computation of current densities, recombination rates,
 *  mobilities and their derivatives
 */
void 
  TWONcommonTerms(TWOdevice *pDevice, BOOLEAN currentOnly, 
                  BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOedge *pEdge;
  TWOnode *pNode;
  int index, eIndex;
  int nextIndex;			/* index of node to find next element */
  double psi1, psi2, refPsi, nC, nP1;
  double dPsiN;
  double bPsiN, dbPsiN, bMPsiN, dbMPsiN;
  double muN, dMuN, rDx, rDy;
  double psi, nConc = 0.0, pConc = 0.0;
  double cnAug, cpAug;
  double eSurf = 0.0;			/* For channel mobilities */
  double qInt = 0.0;
  TWOchannel *pCh;
  
  /* evaluate all node (including recombination) and edge quantities */
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    refPsi = pElem->matlInfo->refPsi;
    cnAug = pElem->matlInfo->cAug[ELEC];
    cpAug = pElem->matlInfo->cAug[HOLE];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) { 
	pNode = pElem->pNodes[ index ]; 
	if ( pNode->nodeType != CONTACT ) { 
	  psi = pDevice->dcSolution[ pNode->psiEqn ];
	  if ( pElem->elemType == SEMICON ) { 
	    nConc = pDevice->dcSolution[ pNode->nEqn ];
            pConc = pNode->nie * exp( - psi + refPsi );
	    if ( Srh ) {
	      recomb(nConc, pConc, 
		     pNode->tn, pNode->tp, cnAug, cpAug, pNode->nie, 
		     &pNode->uNet, &pNode->dUdN, &pNode->dUdP);
	    } else {
	      pNode->uNet = 0.0;
	      pNode->dUdN = 0.0;
	      pNode->dUdP = 0.0;
	    }
	  }
	} else {
	  /* a contact node */
	  psi = pNode->psi;
	  if ( pElem->elemType == SEMICON ) { 
	    nConc = pNode->nConc;
	    pConc = pNode->pConc;
	  }
	}
	
	/* store info in the state tables */
	pDevice->devState0 [pNode->nodePsi] = psi;
	if ( pElem->elemType == SEMICON ) {
	  pDevice->devState0 [pNode->nodeN] = nConc;
	  pDevice->devState0 [pNode->nodeP] = pConc;
	  if ( tranAnalysis && pNode->nodeType != CONTACT ) {
	    pNode->dNdT = integrate( pDevice->devStates, info, pNode->nodeN );
	  }
	}
      }
    }
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalEdges[ index ] ) { 
	pEdge = pElem->pEdges[ index ];
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  psi1 = pDevice->dcSolution[pNode->psiEqn]; 
	} else {
	  psi1 = pNode->psi;
	}
	pNode = pElem->pNodes[ (index + 1) % 4 ];
	if ( pNode->nodeType != CONTACT ) {
	  psi2 = pDevice->dcSolution[pNode->psiEqn]; 
	} else {
	  psi2 = pNode->psi;
	}
	if ( index <= 1 ) {
	  pEdge->dPsi = psi2 - psi1;
	} else {
	  pEdge->dPsi = psi1 - psi2;
	}
	pDevice->devState0 [pEdge->edgeDpsi] = pEdge->dPsi;
	
	if ( pElem->elemType == SEMICON ) {
	  /* Calculate weighted driving forces - wdfn & wdfp for the edge */
	  dPsiN = pEdge->dPsi + pEdge->dCBand;
	  bernoulli( dPsiN, &bPsiN, &dbPsiN,
		    &bMPsiN, &dbMPsiN, !currentOnly );
	  if ( index <= 1 ) {
	    nC = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeN);
	    nP1 = *(pDevice->devState0 + pElem->pNodes[ index+1 ]->nodeN);
	  } else {
	    nC = *(pDevice->devState0 + pElem->pNodes[(index+1)%4]->nodeN);
	    nP1 = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeN);
	  }
	  pEdge->wdfn = bPsiN * nP1 - bMPsiN * nC;
	  pEdge->jn = 0.0;
	  if ( !currentOnly ) {
	    pEdge->dWnDpsiP1 = dbPsiN * nP1 - dbMPsiN * nC;
	    pEdge->dWnDn    = - bMPsiN;
	    pEdge->dWnDnP1   = bPsiN;
	    pEdge->dJnDpsiP1 = 0.0;
	    pEdge->dJnDn     = 0.0;
	    pEdge->dJnDnP1   = 0.0;
	  }
	}
      }
    }
  }
  
  /* DAG: calculate mobilities for channel elems */
  if ( SurfaceMobility ) {
    for ( pCh = pDevice->pChannel;
	 pCh != NULL; pCh = pCh->next ) {
      pElem = pCh->pNElem;
      switch (pCh->type) {
      case 0:
	eSurf = - 0.5 * (pElem->pLeftEdge->dPsi + pElem->pRightEdge->dPsi )
	  * pElem->epsRel / pElem->dy;
	qInt = 0.5 * pElem->pBotEdge->qf;
	break;
      case 1:
	eSurf = - 0.5 * (pElem->pTopEdge->dPsi + pElem->pBotEdge->dPsi )
	  * pElem->epsRel / pElem->dx;
	qInt = 0.5 * pElem->pLeftEdge->qf;
	break;
      case 2:
	eSurf = - 0.5 * (pElem->pLeftEdge->dPsi + pElem->pRightEdge->dPsi )
	  * pElem->epsRel / pElem->dy;
	qInt = 0.5 * pElem->pTopEdge->qf;
	break;
      case 3:
	eSurf = - 0.5 * (pElem->pTopEdge->dPsi + pElem->pBotEdge->dPsi )
	  * pElem->epsRel / pElem->dx;
	qInt = 0.5 * pElem->pRightEdge->qf;
	break;
      }
      eSurf += qInt;
      pElem = pCh->pSeed;
      nextIndex = (pCh->type + 2)%4;
      while (pElem && pElem->channel == pCh->id) {
	TWONmobility( pElem, eSurf );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCH != NULL */
  } /* endif SurfaceMobility */
  
  /* calculate the current densities assuming mobility value depend on Ex,Ey*/
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    
    rDx = 1.0 / pElem->dx;
    rDy = 1.0 / pElem->dy;
    for ( index = 0; index <= 3; index++ ) {
      pEdge = pElem->pEdges[ index ];
      /* calculate conductive currents */
      if ( pElem->elemType == SEMICON ) {
	/* get mobility for this edge */
	if ( !pElem->channel ) {
	  /* Calculate mobility for non-channel elements */
	  muN = pElem->mun0;
	  dMuN = 0.0;
	  dPsiN = pEdge->dPsi + pEdge->dCBand;
	  if ( index%2 == 0 ) {
	    MOBfieldDep( pElem->matlInfo, ELEC, - dPsiN * rDx, &muN, &dMuN );
	  } else {
	    MOBfieldDep( pElem->matlInfo, ELEC, - dPsiN * rDy, &muN, &dMuN );
	  }
	} else {
	  /* Retrieve previously calculated value. */
	  muN = pElem->mun;
	  dMuN = 0.0;
	}
	switch ( index ) {
	case 0:
	  muN *= pEdge->kPos * rDx;
	  dMuN *= pEdge->kPos * rDx * rDx;
	  break;
	case 1:
	  muN *= pEdge->kNeg * rDy;
	  dMuN *= pEdge->kNeg * rDy * rDy;
	  break;
	case 2:
	  muN *= pEdge->kNeg * rDx;
	  dMuN *= pEdge->kNeg * rDx * rDx;
	  break;
	case 3:
	  muN *= pEdge->kPos * rDy;
	  dMuN *= pEdge->kPos * rDy * rDy;
	  break;
	}
	/* Now that the mobility for this edge is known, do current */
	pEdge->jn += muN * pEdge->wdfn;
	if ( !currentOnly ) {
	  pEdge->dJnDpsiP1 += muN * pEdge->dWnDpsiP1;
	  pEdge->dJnDn += muN * pEdge->dWnDn;
	  pEdge->dJnDnP1 += muN * pEdge->dWnDnP1;
	  if ( MobDeriv && (!pElem->channel) ) {
	    pEdge->dJnDpsiP1 -= dMuN * pEdge->wdfn;
	  }
	}
      }
      /* calculate displacement current only once */
      if ( pElem->evalEdges[ index ] ) { 
	if ( tranAnalysis ) {
	  if ( index == 0 || index == 2 ) {
	    /* horizontal edges */
	    pEdge->jd = -integrate(pDevice->devStates, info,
				   pEdge->edgeDpsi) * rDx;
	  } else {
	    /* vertical edges */
	    pEdge->jd = -integrate(pDevice->devStates, info,
				   pEdge->edgeDpsi) * rDy;
	  }
	}
      }
    }
  }
}
