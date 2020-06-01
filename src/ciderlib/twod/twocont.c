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
 * setup matrix pointers to Jacobian values and 
 * store direct pointers with the nodes
 */

void 
  TWO_jacBuild(TWOdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOchannel *pCh;
  int eIndex, nIndex;
  int nextIndex;			/* index of node to find next element */
  int psiEqn, nEqn, pEqn;		/* scratch for deref'd eqn numbers */
  int psiEqnTL = 0, nEqnTL = 0, pEqnTL = 0;
  int psiEqnTR = 0, nEqnTR = 0, pEqnTR = 0;
  int psiEqnBR = 0, nEqnBR = 0, pEqnBR = 0;
  int psiEqnBL = 0, nEqnBL = 0, pEqnBL = 0;
  int psiEqnInM = 0, psiEqnInP = 0;	/* scratch for deref'd surface eqns */
  int psiEqnOxM = 0, psiEqnOxP = 0;	/* M= more negative, P= more positive */
  
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
        pNode->fPN = SMPmakeEltKLUforCIDER (matrix, pEqn, nEqn) ;
        pNode->fPNBinding = NULL ;
#else
	pNode->fPN = SMPmakeElt(matrix, pEqn, nEqn);
#endif

#ifdef KLU
        pNode->fPP = SMPmakeEltKLUforCIDER (matrix, pEqn, pEqn) ;
        pNode->fPPBinding = NULL ;
#else
	pNode->fPP = SMPmakeElt(matrix, pEqn, pEqn);
#endif

      } else {
	nEqn = 0;
	pEqn = 0;
      }
      /* save equation indices */
      switch ( nIndex ) {
      case 0: /* TL Node */
	psiEqnTL = psiEqn;
	nEqnTL = nEqn;
	pEqnTL = pEqn;
	break;
      case 1: /* TR Node */
	psiEqnTR = psiEqn;
	nEqnTR = nEqn;
	pEqnTR = pEqn;
	break;
      case 2: /* BR Node */
	psiEqnBR = psiEqn;
	nEqnBR = nEqn;
	pEqnBR = pEqn;
	break;
      case 3: /* BL Node */
	psiEqnBL = psiEqn;
	nEqnBL = nEqn;
	pEqnBL = pEqn;
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

#ifdef KLU
      pNode->fPPsiiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, psiEqnTR) ;
      pNode->fPPsiiP1Binding = NULL ;
#else
      pNode->fPPsiiP1 = SMPmakeElt(matrix, pEqnTL, psiEqnTR);
#endif

#ifdef KLU
      pNode->fPPiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, pEqnTR) ;
      pNode->fPPiP1Binding = NULL ;
#else
      pNode->fPPiP1 = SMPmakeElt(matrix, pEqnTL, pEqnTR);
#endif

#ifdef KLU
      pNode->fPPsijP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, psiEqnBL) ;
      pNode->fPPsijP1Binding = NULL ;
#else
      pNode->fPPsijP1 = SMPmakeElt(matrix, pEqnTL, psiEqnBL);
#endif

#ifdef KLU
      pNode->fPPjP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, pEqnBL) ;
      pNode->fPPjP1Binding = NULL ;
#else
      pNode->fPPjP1 = SMPmakeElt(matrix, pEqnTL, pEqnBL);
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

#ifdef KLU
        pNode->fPPsiiP1jP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, psiEqnBR) ;
        pNode->fPPsiiP1jP1Binding = NULL ;
#else
        pNode->fPPsiiP1jP1 = SMPmakeElt(matrix, pEqnTL, psiEqnBR);
#endif

#ifdef KLU
        pNode->fPPiP1jP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTL, pEqnBR) ;
        pNode->fPPiP1jP1Binding = NULL ;
#else
        pNode->fPPiP1jP1 = SMPmakeElt(matrix, pEqnTL, pEqnBR);
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

#ifdef KLU
      pNode->fPPsiiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, psiEqnTL) ;
      pNode->fPPsiiM1Binding = NULL ;
#else
      pNode->fPPsiiM1 = SMPmakeElt(matrix, pEqnTR, psiEqnTL);
#endif

#ifdef KLU
      pNode->fPPiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, pEqnTL) ;
      pNode->fPPiM1Binding = NULL ;
#else
      pNode->fPPiM1 = SMPmakeElt(matrix, pEqnTR, pEqnTL);
#endif

#ifdef KLU
      pNode->fPPsijP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, psiEqnBR) ;
      pNode->fPPsijP1Binding = NULL ;
#else
      pNode->fPPsijP1 = SMPmakeElt(matrix, pEqnTR, psiEqnBR);
#endif

#ifdef KLU
      pNode->fPPjP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, pEqnBR) ;
      pNode->fPPjP1Binding = NULL ;
#else
      pNode->fPPjP1 = SMPmakeElt(matrix, pEqnTR, pEqnBR);
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

#ifdef KLU
        pNode->fPPsiiM1jP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, psiEqnBL) ;
        pNode->fPPsiiM1jP1Binding = NULL ;
#else
        pNode->fPPsiiM1jP1 = SMPmakeElt(matrix, pEqnTR, psiEqnBL);
#endif

#ifdef KLU
        pNode->fPPiM1jP1 = SMPmakeEltKLUforCIDER (matrix, pEqnTR, pEqnBL) ;
        pNode->fPPiM1jP1Binding = NULL ;
#else
        pNode->fPPiM1jP1 = SMPmakeElt(matrix, pEqnTR, pEqnBL);
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

#ifdef KLU
      pNode->fPPsiiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, psiEqnBL) ;
      pNode->fPPsiiM1Binding = NULL ;
#else
      pNode->fPPsiiM1 = SMPmakeElt(matrix, pEqnBR, psiEqnBL);
#endif

#ifdef KLU
      pNode->fPPiM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, pEqnBL) ;
      pNode->fPPiM1Binding = NULL ;
#else
      pNode->fPPiM1 = SMPmakeElt(matrix, pEqnBR, pEqnBL);
#endif

#ifdef KLU
      pNode->fPPsijM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, psiEqnTR) ;
      pNode->fPPsijM1Binding = NULL ;
#else
      pNode->fPPsijM1 = SMPmakeElt(matrix, pEqnBR, psiEqnTR);
#endif

#ifdef KLU
      pNode->fPPjM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, pEqnTR) ;
      pNode->fPPjM1Binding = NULL ;
#else
      pNode->fPPjM1 = SMPmakeElt(matrix, pEqnBR, pEqnTR);
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

#ifdef KLU
        pNode->fPPsiiM1jM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, psiEqnTL) ;
        pNode->fPPsiiM1jM1Binding = NULL ;
#else
        pNode->fPPsiiM1jM1 = SMPmakeElt(matrix, pEqnBR, psiEqnTL);
#endif

#ifdef KLU
        pNode->fPPiM1jM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBR, pEqnTL) ;
        pNode->fPPiM1jM1Binding = NULL ;
#else
        pNode->fPPiM1jM1 = SMPmakeElt(matrix, pEqnBR, pEqnTL);
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

#ifdef KLU
      pNode->fPPsiiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, psiEqnBR) ;
      pNode->fPPsiiP1Binding = NULL ;
#else
      pNode->fPPsiiP1 = SMPmakeElt(matrix, pEqnBL, psiEqnBR);
#endif

#ifdef KLU
      pNode->fPPiP1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, pEqnBR) ;
      pNode->fPPiP1Binding = NULL ;
#else
      pNode->fPPiP1 = SMPmakeElt(matrix, pEqnBL, pEqnBR);
#endif

#ifdef KLU
      pNode->fPPsijM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, psiEqnTL) ;
      pNode->fPPsijM1Binding = NULL ;
#else
      pNode->fPPsijM1 = SMPmakeElt(matrix, pEqnBL, psiEqnTL);
#endif

#ifdef KLU
      pNode->fPPjM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, pEqnTL) ;
      pNode->fPPjM1Binding = NULL ;
#else
      pNode->fPPjM1 = SMPmakeElt(matrix, pEqnBL, pEqnTL);
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

#ifdef KLU
        pNode->fPPsiiP1jM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, psiEqnTR) ;
        pNode->fPPsiiP1jM1Binding = NULL ;
#else
        pNode->fPPsiiP1jM1 = SMPmakeElt(matrix, pEqnBL, psiEqnTR);
#endif

#ifdef KLU
        pNode->fPPiP1jM1 = SMPmakeEltKLUforCIDER (matrix, pEqnBL, pEqnTR) ;
        pNode->fPPiP1jM1Binding = NULL ;
#else
        pNode->fPPiP1jM1 = SMPmakeElt(matrix, pEqnBL, pEqnTR);
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
	  pEqn = pNode->pEqn;
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

#ifdef KLU
              pNode->fPPsiIn = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInM) ;
              pNode->fPPsiInBinding = NULL ;
#else
              pNode->fPPsiIn = SMPmakeElt(matrix, pEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fPPsiInP1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInP) ;
              pNode->fPPsiInP1Binding = NULL ;
#else
              pNode->fPPsiInP1 = SMPmakeElt(matrix, pEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fPPsiOx = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxM) ;
              pNode->fPPsiOxBinding = NULL ;
#else
              pNode->fPPsiOx = SMPmakeElt(matrix, pEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fPPsiOxP1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxP) ;
              pNode->fPPsiOxP1Binding = NULL ;
#else
              pNode->fPPsiOxP1 = SMPmakeElt(matrix, pEqn, psiEqnOxP);
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

#ifdef KLU
              pNode->fPPsiInM1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInM) ;
              pNode->fPPsiInM1Binding = NULL ;
#else
              pNode->fPPsiInM1 = SMPmakeElt(matrix, pEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fPPsiIn = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInP) ;
              pNode->fPPsiInBinding = NULL ;
#else
              pNode->fPPsiIn = SMPmakeElt(matrix, pEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fPPsiOxM1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxM) ;
              pNode->fPPsiOxM1Binding = NULL ;
#else
              pNode->fPPsiOxM1 = SMPmakeElt(matrix, pEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fPPsiOx = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxP) ;
              pNode->fPPsiOxBinding = NULL ;
#else
              pNode->fPPsiOx = SMPmakeElt(matrix, pEqn, psiEqnOxP);
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

#ifdef KLU
              pNode->fPPsiIn = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInM) ;
              pNode->fPPsiInBinding = NULL ;
#else
              pNode->fPPsiIn = SMPmakeElt(matrix, pEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fPPsiInP1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInP) ;
              pNode->fPPsiInP1Binding = NULL ;
#else
              pNode->fPPsiInP1 = SMPmakeElt(matrix, pEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fPPsiOx = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxM) ;
              pNode->fPPsiOxBinding = NULL ;
#else
              pNode->fPPsiOx = SMPmakeElt(matrix, pEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fPPsiOxP1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxP) ;
              pNode->fPPsiOxP1Binding = NULL ;
#else
              pNode->fPPsiOxP1 = SMPmakeElt(matrix, pEqn, psiEqnOxP);
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

#ifdef KLU
              pNode->fPPsiInM1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInM) ;
              pNode->fPPsiInM1Binding = NULL ;
#else
              pNode->fPPsiInM1 = SMPmakeElt(matrix, pEqn, psiEqnInM);
#endif

#ifdef KLU
              pNode->fPPsiIn = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnInP) ;
              pNode->fPPsiInBinding = NULL ;
#else
              pNode->fPPsiIn = SMPmakeElt(matrix, pEqn, psiEqnInP);
#endif

#ifdef KLU
              pNode->fPPsiOxM1 = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxM) ;
              pNode->fPPsiOxM1Binding = NULL ;
#else
              pNode->fPPsiOxM1 = SMPmakeElt(matrix, pEqn, psiEqnOxM);
#endif

#ifdef KLU
              pNode->fPPsiOx = SMPmakeEltKLUforCIDER (matrix, pEqn, psiEqnOxP) ;
              pNode->fPPsiOxBinding = NULL ;
#else
              pNode->fPPsiOx = SMPmakeElt(matrix, pEqn, psiEqnOxP);
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
TWObindCSC (TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOchannel *pCh;
  BindKluElementCOO i, *matched, *BindStruct, *BindStructCSC ;
  int index ;
  size_t nz ;

  int eIndex, nIndex;
  int nextIndex;			/* index of node to find next element */
  int psiEqn, nEqn, pEqn;		/* scratch for deref'd eqn numbers */
  int psiEqnTL = 0, nEqnTL = 0, pEqnTL = 0;
  int psiEqnTR = 0, nEqnTR = 0, pEqnTR = 0;
  int psiEqnBR = 0, nEqnBR = 0, pEqnBR = 0;
  int psiEqnBL = 0, nEqnBL = 0, pEqnBL = 0;
  int psiEqnInM = 0, psiEqnInP = 0;	/* scratch for deref'd surface eqns */
  int psiEqnOxM = 0, psiEqnOxP = 0;	/* M= more negative, P= more positive */

  BindStruct = pDevice->matrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
  nz = pDevice->matrix->SMPkluMatrix->KLUmatrixNZ ;

  BindStructCSC = (BindKluElementCOO *) malloc (nz * sizeof(BindKluElementCOO)) ;
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
	pEqn = pNode->pEqn;
	/* pointers for additional terms */

        CREATE_KLU_BINDING_TABLE_CIDER(fPsiN, fPsiNBinding, psiEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPsiP, fPsiPBinding, psiEqn, pEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNPsi, fNPsiBinding, nEqn, psiEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNN, fNNBinding, nEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNP, fNPBinding, nEqn, pEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsi, fPPsiBinding, pEqn, psiEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPN, fPNBinding, pEqn, nEqn) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPP, fPPBinding, pEqn, pEqn) ;

      } else {
	nEqn = 0;
	pEqn = 0;
      }
      /* save equation indices */
      switch ( nIndex ) {
      case 0: /* TL Node */
	psiEqnTL = psiEqn;
	nEqnTL = nEqn;
	pEqnTL = pEqn;
	break;
      case 1: /* TR Node */
	psiEqnTR = psiEqn;
	nEqnTR = nEqn;
	pEqnTR = pEqn;
	break;
      case 2: /* BR Node */
	psiEqnBR = psiEqn;
	nEqnBR = nEqn;
	pEqnBR = pEqn;
	break;
      case 3: /* BL Node */
	psiEqnBL = psiEqn;
	nEqnBL = nEqn;
	pEqnBL = pEqn;
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
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiP1, fPPsiiP1Binding, pEqnTL, psiEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiP1, fPPiP1Binding, pEqnTL, pEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsijP1, fPPsijP1Binding, pEqnTL, psiEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPjP1, fPPjP1Binding, pEqnTL, pEqnBL) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1jP1, fNPsiiP1jP1Binding, nEqnTL, psiEqnBR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1jP1, fNNiP1jP1Binding, nEqnTL, nEqnBR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiP1jP1, fPPsiiP1jP1Binding, pEqnTL, psiEqnBR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPiP1jP1, fPPiP1jP1Binding, pEqnTL, pEqnBR) ;

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
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiM1, fPPsiiM1Binding, pEqnTR, psiEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiM1, fPPiM1Binding, pEqnTR, pEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsijP1, fPPsijP1Binding, pEqnTR, psiEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPjP1, fPPjP1Binding, pEqnTR, pEqnBR) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1jP1, fNPsiiM1jP1Binding, nEqnTR, psiEqnBL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1jP1, fNNiM1jP1Binding, nEqnTR, nEqnBL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiM1jP1, fPPsiiM1jP1Binding, pEqnTR, psiEqnBL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPiM1jP1, fPPiM1jP1Binding, pEqnTR, pEqnBL) ;

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
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiM1, fPPsiiM1Binding, pEqnBR, psiEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiM1, fPPiM1Binding, pEqnBR, pEqnBL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsijM1, fPPsijM1Binding, pEqnBR, psiEqnTR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPjM1, fPPjM1Binding, pEqnBR, pEqnTR) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiM1jM1, fNPsiiM1jM1Binding, nEqnBR, psiEqnTL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiM1jM1, fNNiM1jM1Binding, nEqnBR, nEqnTL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiM1jM1, fPPsiiM1jM1Binding, pEqnBR, psiEqnTL) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPiM1jM1, fPPiM1jM1Binding, pEqnBR, pEqnTL) ;

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
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiP1, fPPsiiP1Binding, pEqnBL, psiEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPiP1, fPPiP1Binding, pEqnBL, pEqnBR) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPsijM1, fPPsijM1Binding, pEqnBL, psiEqnTL) ;
      CREATE_KLU_BINDING_TABLE_CIDER(fPPjM1, fPPjM1Binding, pEqnBL, pEqnTL) ;

      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {

        CREATE_KLU_BINDING_TABLE_CIDER(fNPsiiP1jM1, fNPsiiP1jM1Binding, nEqnBL, psiEqnTR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fNNiP1jM1, fNNiP1jM1Binding, nEqnBL, nEqnTR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPsiiP1jM1, fPPsiiP1jM1Binding, pEqnBL, psiEqnTR) ;
        CREATE_KLU_BINDING_TABLE_CIDER(fPPiP1jM1, fPPiP1jM1Binding, pEqnBL, pEqnTR) ;

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
	  pEqn = pNode->pEqn;
	  if ( pCh->type % 2 == 0 ) { /* Vertical Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInP1, fNPsiInP1Binding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxP1, fNPsiOxP1Binding, nEqn, psiEqnOxP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiIn, fPPsiInBinding, pEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiInP1, fPPsiInP1Binding, pEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOx, fPPsiOxBinding, pEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOxP1, fPPsiOxP1Binding, pEqn, psiEqnOxP) ;

	    } else { /* Right Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInM1, fNPsiInM1Binding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxM1, fNPsiOxM1Binding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiInM1, fPPsiInM1Binding, pEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiIn, fPPsiInBinding, pEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOxM1, fPPsiOxM1Binding, pEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOx, fPPsiOxBinding, pEqn, psiEqnOxP) ;

	    }
	  } else { /* Horizontal Slice */
	    if ( nIndex <= 1 ) { /* Top Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInP1, fNPsiInP1Binding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxP1, fNPsiOxP1Binding, nEqn, psiEqnOxP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiIn, fPPsiInBinding, pEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiInP1, fPPsiInP1Binding, pEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOx, fPPsiOxBinding, pEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOxP1, fPPsiOxP1Binding, pEqn, psiEqnOxP) ;

	    } else { /* Bottom Side */

              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiInM1, fNPsiInM1Binding, nEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiIn, fNPsiInBinding, nEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOxM1, fNPsiOxM1Binding, nEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fNPsiOx, fNPsiOxBinding, nEqn, psiEqnOxP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiInM1, fPPsiInM1Binding, pEqn, psiEqnInM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiIn, fPPsiInBinding, pEqn, psiEqnInP) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOxM1, fPPsiOxM1Binding, pEqn, psiEqnOxM) ;
              CREATE_KLU_BINDING_TABLE_CIDER(fPPsiOx, fPPsiOxBinding, pEqn, psiEqnOxP) ;

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
  TWO_sysLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
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
  double rhsN, rhsP;
  double generation;
  double nConc, pConc;
  double perTime = 0.0;
  
  /* first compute the currents and derivatives */
  TWO_commonTerms( pDevice, FALSE, tranAnalysis, info );
  
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
	  *(pNode->fPsiN) += dxdy;
	  *(pNode->fPsiP) -= dxdy;
	  *(pNode->fNPsi) -= dy * pHEdge->dJnDpsiP1 + 
	    dx * pVEdge->dJnDpsiP1;
	  *(pNode->fPPsi) -= dy * pHEdge->dJpDpsiP1 +
	    dx * pVEdge->dJpDpsiP1;
	  
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];
	  pRhs[ pNode->psiEqn ] += dxdy * (pNode->netConc + pConc - nConc);
	  
	  /* Handle generation terms */
	  *(pNode->fNN) -= dxdy * pNode->dUdN;
	  *(pNode->fNP) -= dxdy * pNode->dUdP;
	  *(pNode->fPP) += dxdy * pNode->dUdP;
	  *(pNode->fPN) += dxdy * pNode->dUdN;
	  rhsN = - dxdy * pNode->uNet;
	  rhsP =   dxdy * pNode->uNet;
	  if ( AvalancheGen ) {
	    generation = TWOavalanche( pElem, pNode );
	    rhsN += dxdy * generation;
	    rhsP -= dxdy * generation;
	  }
	  pRhs[ pNode->nEqn ] -= rhsN;
	  pRhs[ pNode->pEqn ] -= rhsP;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
	    *(pNode->fNN) -= dxdy * perTime;
	    *(pNode->fPP) += dxdy * perTime;
	    pRhs[ pNode->nEqn ] += dxdy * pNode->dNdT;
	    pRhs[ pNode->pEqn ] -= dxdy * pNode->dPdT;
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
	pRhs[ pNode->pEqn ] -= dy * pTEdge->jp + dx * pLEdge->jp;
	*(pNode->fNN) += dy * pTEdge->dJnDn + dx * pLEdge->dJnDn;
	*(pNode->fPP) += dy * pTEdge->dJpDp + dx * pLEdge->dJpDp;
	*(pNode->fNPsiiP1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pTEdge->dJnDnP1;
	*(pNode->fPPsiiP1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pTEdge->dJpDpP1;
	*(pNode->fNPsijP1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pLEdge->dJnDnP1;
	*(pNode->fPPsijP1) += dx * pLEdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pLEdge->dJpDpP1;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pTEdge->jn + dx * pREdge->jn;
	pRhs[ pNode->pEqn ] -= -dy * pTEdge->jp + dx * pREdge->jp;
	*(pNode->fNN) += -dy * pTEdge->dJnDnP1 + dx * pREdge->dJnDn;
	*(pNode->fPP) += -dy * pTEdge->dJpDpP1 + dx * pREdge->dJpDp;
	*(pNode->fNPsiiM1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pTEdge->dJnDn;
	*(pNode->fPPsiiM1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pTEdge->dJpDp;
	*(pNode->fNPsijP1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pREdge->dJnDnP1;
	*(pNode->fPPsijP1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pREdge->dJpDpP1;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pBEdge->jn - dx * pREdge->jn;
	pRhs[ pNode->pEqn ] -= -dy * pBEdge->jp - dx * pREdge->jp;
	*(pNode->fNN) += -dy * pBEdge->dJnDnP1 - dx * pREdge->dJnDnP1;
	*(pNode->fPP) += -dy * pBEdge->dJpDpP1 - dx * pREdge->dJpDpP1;
	*(pNode->fNPsiiM1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pBEdge->dJnDn;
	*(pNode->fPPsiiM1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pBEdge->dJpDp;
	*(pNode->fNPsijM1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pREdge->dJnDn;
	*(pNode->fPPsijM1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjM1) -= dx * pREdge->dJpDp;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pBEdge->jn - dx * pLEdge->jn;
	pRhs[ pNode->pEqn ] -= dy * pBEdge->jp - dx * pLEdge->jp;
	*(pNode->fNN) += dy * pBEdge->dJnDn - dx * pLEdge->dJnDnP1;
	*(pNode->fPP) += dy * pBEdge->dJpDp - dx * pLEdge->dJpDpP1;
	*(pNode->fNPsiiP1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pBEdge->dJnDnP1;
	*(pNode->fPPsiiP1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pBEdge->dJpDpP1;
	*(pNode->fNPsijM1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pLEdge->dJnDn;
	*(pNode->fPPsijM1) += dx * pLEdge->dJpDpsiP1;
	*(pNode->fPPjM1) -= dx * pLEdge->dJpDp;
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
	TWO_mobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}


/* this function used only for direct method ac analysis 
   Used to load only the dc Jacobian matrix. Rhs is unaffected
   */

void 
  TWO_jacLoad(TWOdevice *pDevice)
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
  
  /* first compute the currents and derivatives */
  TWO_commonTerms( pDevice, FALSE, FALSE, NULL );
  
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
	  *(pNode->fPsiN) += dxdy;
	  *(pNode->fPsiP) -= dxdy;
	  *(pNode->fNPsi) -= dy * pHEdge->dJnDpsiP1 + 
	    dx * pVEdge->dJnDpsiP1;
	  *(pNode->fPPsi) -= dy * pHEdge->dJpDpsiP1 +
	    dx * pVEdge->dJpDpsiP1;
	  
	  /* Handle generation terms */
	  *(pNode->fNN) -= dxdy * pNode->dUdN;
	  *(pNode->fNP) -= dxdy * pNode->dUdP;
	  *(pNode->fPP) += dxdy * pNode->dUdP;
	  *(pNode->fPN) += dxdy * pNode->dUdN;
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
	*(pNode->fPP) += dy * pTEdge->dJpDp + dx * pLEdge->dJpDp;
	*(pNode->fNPsiiP1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pTEdge->dJnDnP1;
	*(pNode->fPPsiiP1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pTEdge->dJpDpP1;
	*(pNode->fNPsijP1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pLEdge->dJnDnP1;
	*(pNode->fPPsijP1) += dx * pLEdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pLEdge->dJpDpP1;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += -dy * pTEdge->dJnDnP1 + dx * pREdge->dJnDn;
	*(pNode->fPP) += -dy * pTEdge->dJpDpP1 + dx * pREdge->dJpDp;
	*(pNode->fNPsiiM1) += dy * pTEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pTEdge->dJnDn;
	*(pNode->fPPsiiM1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pTEdge->dJpDp;
	*(pNode->fNPsijP1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjP1) += dx * pREdge->dJnDnP1;
	*(pNode->fPPsijP1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pREdge->dJpDpP1;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += -dy * pBEdge->dJnDnP1 - dx * pREdge->dJnDnP1;
	*(pNode->fPP) += -dy * pBEdge->dJpDpP1 - dx * pREdge->dJpDpP1;
	*(pNode->fNPsiiM1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiM1) -= dy * pBEdge->dJnDn;
	*(pNode->fPPsiiM1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pBEdge->dJpDp;
	*(pNode->fNPsijM1) += dx * pREdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pREdge->dJnDn;
	*(pNode->fPPsijM1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjM1) -= dx * pREdge->dJpDp;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fNN) += dy * pBEdge->dJnDn - dx * pLEdge->dJnDnP1;
	*(pNode->fPP) += dy * pBEdge->dJpDp - dx * pLEdge->dJpDpP1;
	*(pNode->fNPsiiP1) += dy * pBEdge->dJnDpsiP1;
	*(pNode->fNNiP1) += dy * pBEdge->dJnDnP1;
	*(pNode->fPPsiiP1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pBEdge->dJpDpP1;
	*(pNode->fNPsijM1) += dx * pLEdge->dJnDpsiP1;
	*(pNode->fNNjM1) -= dx * pLEdge->dJnDn;
	*(pNode->fPPsijM1) += dx * pLEdge->dJpDpsiP1;
	*(pNode->fPPjM1) -= dx * pLEdge->dJpDp;
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
	TWO_mobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}

/* load only the Rhs vector */
void 
  TWO_rhsLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dx, dy, dxdy, dyOverDx, dxOverDy;
  double dPsiT, dPsiB, dPsiL, dPsiR;
  double rhsN, rhsP;
  double generation;
  double nConc, pConc;
  double perTime;
  
  /* first compute the currents */
  TWO_commonTerms( pDevice, TRUE, tranAnalysis, info );
  
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
	  rhsP =   dxdy * pNode->uNet;
	  if ( AvalancheGen ) {
	    generation = TWOavalanche( pElem, pNode );
	    rhsN += dxdy * generation;
	    rhsP -= dxdy * generation;
	  }
	  pRhs[ pNode->nEqn ] -= rhsN;
	  pRhs[ pNode->pEqn ] -= rhsP;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
	    pRhs[ pNode->nEqn ] += dxdy * pNode->dNdT;
	    pRhs[ pNode->pEqn ] -= dxdy * pNode->dPdT;
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
	pRhs[ pNode->pEqn ] -= dy * pTEdge->jp + dx * pLEdge->jp;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pTEdge->jn + dx * pREdge->jn;
	pRhs[ pNode->pEqn ] -= -dy * pTEdge->jp + dx * pREdge->jp;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= -dy * pBEdge->jn - dx * pREdge->jn;
	pRhs[ pNode->pEqn ] -= -dy * pBEdge->jp - dx * pREdge->jp;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->nEqn ] -= dy * pBEdge->jn - dx * pLEdge->jn;
	pRhs[ pNode->pEqn ] -= dy * pBEdge->jp - dx * pLEdge->jp;
      }
    }
  }
}

/*
 * computation of current densities, recombination rates,
 *  mobilities and their derivatives
 */
void 
  TWO_commonTerms(TWOdevice *pDevice, BOOLEAN currentOnly, 
                  BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOedge *pEdge;
  TWOnode *pNode;
  int index, eIndex;
  int nextIndex;			/* index of node to find next element */
  double psi1, psi2, nC, pC, nP1, pP1;
  double dPsiN, dPsiP;
  double bPsiN, dbPsiN, bMPsiN, dbMPsiN;
  double bPsiP, dbPsiP, bMPsiP, dbMPsiP;
  double muN, dMuN, muP, dMuP, rDx, rDy;
  double psi, nConc = 0.0, pConc = 0.0;
  double cnAug, cpAug;
  double eSurf = 0.0;			/* For channel mobilities */
  double qInt = 0.0;
  TWOchannel *pCh;
  
  /* evaluate all node (including recombination) and edge quantities */
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    cnAug = pElem->matlInfo->cAug[ELEC];
    cpAug = pElem->matlInfo->cAug[HOLE];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) { 
	pNode = pElem->pNodes[ index ]; 
	if ( pNode->nodeType != CONTACT ) { 
	  psi = pDevice->dcSolution[ pNode->psiEqn ];
	  if ( pElem->elemType == SEMICON ) { 
	    nConc = pDevice->dcSolution[ pNode->nEqn ];
	    pConc = pDevice->dcSolution[ pNode->pEqn ];
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
	  if ( tranAnalysis && (pNode->nodeType != CONTACT) ) {
	    pNode->dNdT = integrate( pDevice->devStates, info, pNode->nodeN );
	    pNode->dPdT = integrate( pDevice->devStates, info, pNode->nodeP );
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
	  dPsiP = pEdge->dPsi - pEdge->dVBand;
	  bernoulli( dPsiN, &bPsiN, &dbPsiN,
		    &bMPsiN, &dbMPsiN, !currentOnly );
	  bernoulli( dPsiP, &bPsiP, &dbPsiP,
		    &bMPsiP, &dbMPsiP, !currentOnly);
	  if ( index <= 1 ) {
	    nC = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeN);
	    nP1 = *(pDevice->devState0 + pElem->pNodes[ index+1 ]->nodeN);
	    pC = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeP);
	    pP1 = *(pDevice->devState0 + pElem->pNodes[ index+1 ]->nodeP);
	  } else {
	    nC = *(pDevice->devState0 + pElem->pNodes[(index+1)%4]->nodeN);
	    nP1 = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeN);
	    pC = *(pDevice->devState0 + pElem->pNodes[(index+1)%4]->nodeP);
	    pP1 = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeP);
	  }
	  pEdge->wdfn = bPsiN * nP1 - bMPsiN * nC;
	  pEdge->wdfp = bPsiP * pC - bMPsiP * pP1;
	  pEdge->jn = 0.0;
	  pEdge->jp = 0.0;
	  if ( !currentOnly ) {
	    pEdge->dWnDpsiP1 = dbPsiN * nP1 - dbMPsiN * nC;
	    pEdge->dWnDn    = - bMPsiN;
	    pEdge->dWnDnP1   = bPsiN;
	    pEdge->dWpDpsiP1 = dbPsiP * pC - dbMPsiP * pP1;
	    pEdge->dWpDp     = bPsiP;
	    pEdge->dWpDpP1   = - bMPsiP;
	    pEdge->dJnDpsiP1 = 0.0;
	    pEdge->dJnDn     = 0.0;
	    pEdge->dJnDnP1   = 0.0;
	    pEdge->dJpDpsiP1 = 0.0;
	    pEdge->dJpDp     = 0.0;
	    pEdge->dJpDpP1   = 0.0;
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
	TWO_mobility( pElem, eSurf );
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
	  muP = pElem->mup0;
	  dMuP = 0.0;
	  dPsiN = pEdge->dPsi + pEdge->dCBand;
	  dPsiP = pEdge->dPsi - pEdge->dVBand;
	  if ( index%2 == 0 ) {
	    MOBfieldDep( pElem->matlInfo, ELEC, - dPsiN * rDx, &muN, &dMuN );
	    MOBfieldDep( pElem->matlInfo, HOLE, - dPsiP * rDx, &muP, &dMuP );
	  } else {
	    MOBfieldDep( pElem->matlInfo, ELEC, - dPsiN * rDy, &muN, &dMuN );
	    MOBfieldDep( pElem->matlInfo, HOLE, - dPsiP * rDy, &muP, &dMuP );
	  }
	} else {
	  /* Retrieve previously calculated value. */
	  muN = pElem->mun;
	  dMuN = 0.0;
	  muP = pElem->mup;
	  dMuP = 0.0;
	}
	switch ( index ) {
	case 0:
	  muN *= pEdge->kPos * rDx;
	  dMuN *= pEdge->kPos * rDx * rDx;
	  muP *= pEdge->kPos * rDx;
	  dMuP *= pEdge->kPos * rDx * rDx;
	  break;
	case 1:
	  muN *= pEdge->kNeg * rDy;
	  dMuN *= pEdge->kNeg * rDy * rDy;
	  muP *= pEdge->kNeg * rDy;
	  dMuP *= pEdge->kNeg * rDy * rDy;
	  break;
	case 2:
	  muN *= pEdge->kNeg * rDx;
	  dMuN *= pEdge->kNeg * rDx * rDx;
	  muP *= pEdge->kNeg * rDx;
	  dMuP *= pEdge->kNeg * rDx * rDx;
	  break;
	case 3:
	  muN *= pEdge->kPos * rDy;
	  dMuN *= pEdge->kPos * rDy * rDy;
	  muP *= pEdge->kPos * rDy;
	  dMuP *= pEdge->kPos * rDy * rDy;
	  break;
	}
	/* Now that the mobility for this edge is known, do current */
	pEdge->jn += muN * pEdge->wdfn;
	pEdge->jp += muP * pEdge->wdfp;
	if ( !currentOnly ) {
	  pEdge->dJnDpsiP1 += muN * pEdge->dWnDpsiP1;
	  pEdge->dJnDn += muN * pEdge->dWnDn;
	  pEdge->dJnDnP1 += muN * pEdge->dWnDnP1;
	  pEdge->dJpDpsiP1 += muP * pEdge->dWpDpsiP1;
	  pEdge->dJpDp += muP * pEdge->dWpDp;
	  pEdge->dJpDpP1 += muP * pEdge->dWpDpP1;
	  if ( MobDeriv && (!pElem->channel) ) {
	    pEdge->dJnDpsiP1 -= dMuN * pEdge->wdfn;
	    pEdge->dJpDpsiP1 -= dMuP * pEdge->wdfp;
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
