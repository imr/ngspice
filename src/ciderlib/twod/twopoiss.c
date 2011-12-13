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
#include "ngspice/spmatrix.h"
#include "twoddefs.h"
#include "twodext.h"


/* functions to setup and solve the 2D poisson equation */

void 
TWOQjacBuild(TWOdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  TWOelem *pElem;
  TWOnode *pNode, *pNode1;
  int eIndex, nIndex;

  /* set up matrix pointers */
  /* establish main diagonal first */
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for ( nIndex = 0; nIndex <= 3; nIndex++ ) {
      if ( pElem->evalNodes[ nIndex ] ) { 
	pNode = pElem->pNodes[ nIndex ]; 
	pNode->fPsiPsi = spGetElement( matrix, pNode->poiEqn, pNode->poiEqn );
      }
    }
  }
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];

    pNode = pElem->pTLNode;
    pNode1 = pElem->pTRNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pBLNode;
    pNode->fPsiPsijP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pTRNode;
    pNode1 = pElem->pTLNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pBRNode;
    pNode->fPsiPsijP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pBRNode;
    pNode1 = pElem->pBLNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pTRNode;
    pNode->fPsiPsijM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pBLNode;
    pNode1 = pElem->pBRNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pTLNode;
    pNode->fPsiPsijM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
  }
  /*
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];

    pNode = pElem->pTLNode;
    pNode->fPsiPsi = spGetElement( matrix, pNode->poiEqn, pNode->poiEqn );
    pNode1 = pElem->pTRNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pBLNode;
    pNode->fPsiPsijP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pTRNode;
    pNode->fPsiPsi = spGetElement( matrix, pNode->poiEqn, pNode->poiEqn );
    pNode1 = pElem->pTLNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pBRNode;
    pNode->fPsiPsijP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pBRNode;
    pNode->fPsiPsi = spGetElement( matrix, pNode->poiEqn, pNode->poiEqn );
    pNode1 = pElem->pBLNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pTRNode;
    pNode->fPsiPsijM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode = pElem->pBLNode;
    pNode->fPsiPsi = spGetElement( matrix, pNode->poiEqn, pNode->poiEqn );
    pNode1 = pElem->pBRNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
    pNode1 = pElem->pTLNode;
    pNode->fPsiPsijM1 = spGetElement(matrix, pNode->poiEqn, pNode1->poiEqn );
  }
  */
}

void 
TWOQsysLoad(TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dyOverDx, dxOverDy, dPsiT, dPsiB, dPsiL, dPsiR;
  
  TWOQcommonTerms( pDevice );

  /* zero the rhs vector */
  for ( index = 1 ; index <= pDevice->numEqns ; index++ ) {
    pRhs[ index ] = 0.0;
  }

  /* zero the matrix */
  spClear( pDevice->matrix );

  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];

    dxOverDy = 0.5 * pElem->epsRel * pElem->dxOverDy;
    dyOverDx = 0.5 * pElem->epsRel * pElem->dyOverDx;
    dPsiT = pElem->pTopEdge->dPsi;
    dPsiB = pElem->pBotEdge->dPsi;
    dPsiL = pElem->pLeftEdge->dPsi;
    dPsiR = pElem->pRightEdge->dPsi;

    /* load for all i,j */
    for ( index = 0; index <= 3; index++ ) {
      pNode = pElem->pNodes[ index ];
      if ( pNode->nodeType != CONTACT ) {
	*(pNode->fPsiPsi) += dyOverDx + dxOverDy;
	if ( index <= 1 ) {
	  pHEdge = pElem->pTopEdge;
	} else {
	  pHEdge = pElem->pBotEdge;
	}
	if ( index == 0 || index == 3 ) {
	  pVEdge = pElem->pLeftEdge;
	} else {
	  pVEdge = pElem->pRightEdge;
	}
	/* add surface state charges */
	pRhs[ pNode->poiEqn ] += 0.5 * pElem->dx * pHEdge->qf;
	pRhs[ pNode->poiEqn ] += 0.5 * pElem->dy * pVEdge->qf;
	if ( pElem->elemType == SEMICON ) {
	  *(pNode->fPsiPsi) += 0.25 * pElem->dx * pElem->dy *
	      (pNode->nConc + pNode->pConc);
	  pRhs[ pNode->poiEqn ] += 0.25 * pElem->dx * pElem->dy *
	      (pNode->netConc + pNode->pConc - pNode->nConc);
	}
      }
    }

    pNode = pElem->pTLNode;
    pRhs[ pNode->poiEqn ] -= -dyOverDx * dPsiT - dxOverDy * dPsiL;
    *(pNode->fPsiPsiiP1) -= dyOverDx;
    *(pNode->fPsiPsijP1) -= dxOverDy;

    pNode = pElem->pTRNode;
    pRhs[ pNode->poiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
    *(pNode->fPsiPsiiM1) -= dyOverDx;
    *(pNode->fPsiPsijP1) -= dxOverDy;

    pNode = pElem->pBRNode;
    pRhs[ pNode->poiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
    *(pNode->fPsiPsiiM1) -= dyOverDx;
    *(pNode->fPsiPsijM1) -= dxOverDy;

    pNode = pElem->pBLNode;
    pRhs[ pNode->poiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
    *(pNode->fPsiPsiiP1) -= dyOverDx;
    *(pNode->fPsiPsijM1) -= dxOverDy;
  }
}


void 
TWOQrhsLoad(TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dyOverDx, dxOverDy, dPsiT, dPsiB, dPsiL, dPsiR;

  TWOQcommonTerms( pDevice );

  /* zero the rhs vector */
  for ( index = 1 ; index <= pDevice->numEqns ; index++ ) {
    pRhs[ index ] = 0.0;
  }

  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];

    dxOverDy = 0.5 * pElem->epsRel * pElem->dxOverDy;
    dyOverDx = 0.5 * pElem->epsRel * pElem->dyOverDx;
    dPsiT = pElem->pTopEdge->dPsi;
    dPsiB = pElem->pBotEdge->dPsi;
    dPsiL = pElem->pLeftEdge->dPsi;
    dPsiR = pElem->pRightEdge->dPsi;

    /* load nodal terms */
    for ( index = 0; index <= 3; index++ ) {
      pNode = pElem->pNodes[ index ];
      if ( (pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON) ) {
	pRhs[ pNode->poiEqn ] += 0.25 * pElem->dx * pElem->dy *
	  (pNode->netConc + pNode->pConc - pNode->nConc);
      }
      if ( index <= 1 ) {
	pHEdge = pElem->pTopEdge;
      } else {
	pHEdge = pElem->pBotEdge;
      }
      if ( index == 0 || index == 3 ) {
	pVEdge = pElem->pLeftEdge;
      } else {
	pVEdge = pElem->pRightEdge;
      }
      /* add surface state charges */
      pRhs[ pNode->poiEqn ] += 0.5 * pElem->dx * pHEdge->qf;
      pRhs[ pNode->poiEqn ] += 0.5 * pElem->dy * pVEdge->qf;
    }

    /* load edge terms */
    pNode = pElem->pTLNode;
    pRhs[ pNode->poiEqn ] -= -dyOverDx * dPsiT - dxOverDy * dPsiL;

    pNode = pElem->pTRNode;
    pRhs[ pNode->poiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;

    pNode = pElem->pBRNode;
    pRhs[ pNode->poiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;

    pNode = pElem->pBLNode;
    pRhs[ pNode->poiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
  }
}

void 
TWOQcommonTerms(TWOdevice *pDevice)
{
  TWOelem *pElem;
  TWOedge *pEdge;
  TWOnode *pNode;
  int index, eIndex;
  double psi1, psi2, refPsi;

  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    refPsi = pElem->matlInfo->refPsi;
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) { 
	pNode = pElem->pNodes[ index ]; 
	if ( pNode->nodeType != CONTACT ) { 
	  pNode->psi = pDevice->dcSolution[ pNode->poiEqn ];
	  if ( pElem->elemType == SEMICON ) { 
	    pNode->nConc = pNode->nie * exp(   pNode->psi - refPsi );
	    pNode->pConc = pNode->nie * exp( - pNode->psi + refPsi );
	  }
	}
      }
      if ( pElem->evalEdges[ index ] ) { 
	pEdge = pElem->pEdges[ index ];
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  psi1 = pDevice->dcSolution[pNode->poiEqn]; 
	} else {
	  psi1 = pNode->psi;
	}
	pNode = pElem->pNodes[ (index + 1) % 4 ];
	if ( pNode->nodeType != CONTACT ) {
	  psi2 = pDevice->dcSolution[pNode->poiEqn]; 
	} else {
	  psi2 = pNode->psi;
	}
	if ( index <= 1 ) {
	  pEdge->dPsi = psi2 - psi1;
	} else {
	  pEdge->dPsi = psi1 - psi2;
	}
      }
    }
  }
}
