/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Functions for projecting the next solution point for use with the modified 
 * two-level Newton scheme
 */

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


/* Forward Declarations */


void NUMD2project(TWOdevice *pDevice, double delV)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pContact = pDevice->pLastContact;
  double *incVpn, *solution = pDevice->dcSolution;
  double delPsi, delN, delP, newN, newP;
  
  delV = -delV / VNorm;
  /* update the boundary condition for the last contact */
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    pNode->psi += delV;
  }
  
  /* 
   * store the new rhs for computing the incremental quantities
   * with the second to last node. solve the system of equations
   */
  
  if ( ABS(delV) < MIN_DELV ) {
    TWOstoreInitialGuess( pDevice );
    return;
  }
  incVpn = pDevice->dcDeltaSolution;
  storeNewRhs( pDevice, pContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVpn, NULL, NULL );
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) {
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  delPsi = incVpn[ pNode->psiEqn ] * delV;
	  solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	    delN = incVpn[ pNode->nEqn ] * delV;
	    newN = pNode->nConc + delN;
	    if ( newN <= 0.0 ) {
	      solution[ pNode->nEqn ] = guessNewConc( pNode->nConc, delN );
	    }
	    else {
	      solution[ pNode->nEqn ] = newN;
	    }
	  }
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	    delP = incVpn[ pNode->pEqn ] * delV;
	    newP = pNode->pConc + delP;
	    if ( newP <= 0.0 ) {
	      solution[ pNode->pEqn ] = guessNewConc( pNode->pConc, delP );
	    }
	    else {
	      solution[ pNode->pEqn ] = newP;
	    }
	  }
	}
      }
    }
  }
}


void NBJT2project(TWOdevice *pDevice, double delVce, double delVbe)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pColContact = pDevice->pFirstContact;
  TWOcontact *pBaseContact = pDevice->pFirstContact->next;
  double *incVce, *incVbe, *solution = pDevice->dcSolution;
  double delPsi, delN, delP, newN, newP;
  double nConc, pConc;
  
  /* Normalize the voltages for calculations. */
  if ( delVce != 0.0 ) {
    delVce = delVce / VNorm;
    numContactNodes = pColContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pColContact->pNodes[ index ];
      pNode->psi += delVce;
    }
  }
  if ( delVbe != 0.0 ) {
    delVbe = delVbe / VNorm;
    numContactNodes = pBaseContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pBaseContact->pNodes[ index ];
      pNode->psi += delVbe;
    }
  }
  
  /* 
   * store the new rhs for computing the incremental quantities
   * incVce (dcDeltaSolution) and incVbe (copiedSolution) are used to
   * store the incremental quantities associated with Vce and Vbe
   */
  
  /* set incVce = dcDeltaSolution; incVbe = copiedSolution */
  
  if ( ABS( delVce ) > MIN_DELV ) {
    incVce = pDevice->dcDeltaSolution;
    storeNewRhs( pDevice, pColContact );
    spSolve( pDevice->matrix, pDevice->rhs, incVce, NULL, NULL);
    
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if ( pNode->nodeType != CONTACT ) {
	    delPsi = incVce[ pNode->psiEqn ] * delVce;
	    solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	      delN = incVce[ pNode->nEqn ] * delVce;
	      newN = pNode->nConc + delN;
	      if ( newN <= 0.0 ) {
		solution[ pNode->nEqn ] = guessNewConc( pNode->nConc, delN );
	      }
	      else {
		solution[ pNode->nEqn ] = newN;
	      }
	    }
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	      delP = incVce[ pNode->pEqn ] * delVce;
	      newP = pNode->pConc + delP;
	      if ( newP <= 0.0 ) {
		solution[ pNode->pEqn ] = guessNewConc( pNode->pConc, delP );
	      }
	      else {
		solution[ pNode->pEqn ] = newP;
	      }
	    }
	  }
	}
      }
    }
  }
  else {
    TWOstoreInitialGuess( pDevice );
  }
  
  if ( ABS( delVbe ) > MIN_DELV ) {
    incVbe = pDevice->copiedSolution;
    storeNewRhs( pDevice, pBaseContact );
    spSolve( pDevice->matrix, pDevice->rhs, incVbe, NULL, NULL);
    
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if ( pNode->nodeType != CONTACT ) {
	    delPsi = incVbe[ pNode->psiEqn ] * delVbe;
	    solution[ pNode->psiEqn ] += delPsi;
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	      delN = incVbe[ pNode->nEqn ] * delVbe;
	      nConc = solution[ pNode->nEqn ];
	      newN = nConc + delN;
	      if ( newN <= 0.0 ) {
		solution[ pNode->nEqn ] = guessNewConc( nConc, delN );
	      }
	      else {
		solution[ pNode->nEqn ] = newN;
	      }
	    }
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	      delP = incVbe[ pNode->pEqn ] * delVbe;
	      pConc = solution[ pNode->pEqn ];
	      newP = pConc + delP;
	      if ( newP <= 0.0 ) {
		solution[ pNode->pEqn ] = guessNewConc( pConc, delP );
	      }
	      else {
		solution[ pNode->pEqn ] = newP;
	      }
	    }
	  }
	}
      }
    }
  }
}


void NUMOSproject(TWOdevice *pDevice, double delVdb, double delVsb, 
                  double delVgb)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
  double *incVdb, *incVsb, *incVgb, *solution = pDevice->dcSolution;
  double delPsi, delN, delP, newN, newP;
  double nConc, pConc;
  
  /* normalize the voltages for calculations */
  if ( delVdb != 0.0 ) {
    delVdb = delVdb / VNorm;
    numContactNodes = pDContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pDContact->pNodes[ index ];
      pNode->psi += delVdb;
    }
  }
  if ( delVsb != 0.0 ) {
    delVsb = delVsb / VNorm;
    numContactNodes = pSContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pSContact->pNodes[ index ];
      pNode->psi += delVsb;
    }
  }
  if ( delVgb != 0.0 ) {
    delVgb = delVgb / VNorm;
    numContactNodes = pGContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pGContact->pNodes[ index ];
      pNode->psi += delVgb;
    }
  }
  
  /* 
   * store the new rhs for computing the incremental quantities
   * incVdb (dcDeltaSolution), incVsb, incVgb 
   */
  
  if ( ABS( delVdb ) > MIN_DELV ) {
    
    incVdb = pDevice->dcDeltaSolution;
    storeNewRhs( pDevice, pDContact );
    spSolve( pDevice->matrix, pDevice->rhs, incVdb, NULL, NULL);
    
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if ( pNode->nodeType != CONTACT ) {
	    delPsi = incVdb[ pNode->psiEqn ] * delVdb;
	    solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	      delN = incVdb[ pNode->nEqn ] * delVdb;
	      newN = pNode->nConc + delN;
	      if ( newN <= 0.0 ) {
		solution[ pNode->nEqn ] = guessNewConc( pNode->nConc, delN );
	      }
	      else {
		solution[ pNode->nEqn ] = newN;
	      }
	    }
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	      delP = incVdb[ pNode->pEqn ] * delVdb;
	      newP = pNode->pConc + delP;
	      if ( newP <= 0.0 ) {
		solution[ pNode->pEqn ] = guessNewConc( pNode->pConc, delP );
	      }
	      else {
		solution[ pNode->pEqn ] = newP;
	      }
	    }
	  }
	}
      }
    }
  }
  else {
    TWOstoreInitialGuess( pDevice );
  }
  
  if ( ABS( delVsb ) > MIN_DELV ) {
    
    incVsb = pDevice->dcDeltaSolution;
    storeNewRhs( pDevice, pSContact );
    spSolve( pDevice->matrix, pDevice->rhs, incVsb, NULL, NULL);
    
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if ( pNode->nodeType != CONTACT ) {
	    delPsi = incVsb[ pNode->psiEqn ] * delVsb;
	    solution[ pNode->psiEqn ] += delPsi;
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	      delN = incVsb[ pNode->nEqn ] * delVsb;
	      nConc = solution[ pNode->nEqn ];
	      newN = nConc + delN;
	      if ( newN <= 0.0 ) {
		solution[ pNode->nEqn ] = guessNewConc( nConc, delN );
	      }
	      else {
		solution[ pNode->nEqn ] = newN;
	      }
	    }
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	      delP = incVsb[ pNode->pEqn ] * delVsb;
	      pConc = solution[ pNode->pEqn ];
	      newP = pConc + delP;
	      if ( newP <= 0.0 ) {
		solution[ pNode->pEqn ] = guessNewConc( pConc, delP );
	      }
	      else {
		solution[ pNode->pEqn ] = newP;
	      }
	    }
	  }
	}
      }
    }
  }
  
  if ( ABS( delVgb ) > MIN_DELV ) {
    
    incVgb = pDevice->dcDeltaSolution;
    storeNewRhs( pDevice, pGContact );
    spSolve( pDevice->matrix, pDevice->rhs, incVgb, NULL, NULL);
    
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if ( pNode->nodeType != CONTACT ) {
	    delPsi = incVgb[ pNode->psiEqn ] * delVgb;
	    solution[ pNode->psiEqn ] += delPsi;
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	      delN = incVgb[ pNode->nEqn ] * delVgb;
	      nConc = solution[ pNode->nEqn ];
	      newN = nConc + delN;
	      if ( newN <= 0.0 ) {
		solution[ pNode->nEqn ] = guessNewConc( nConc, delN );
	      }
	      else {
		solution[ pNode->nEqn ] = newN;
	      }
	    }
	    if ( pElem->elemType == SEMICON
		&& (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	      delP = incVgb[ pNode->pEqn ] * delVgb;
	      pConc = solution[ pNode->pEqn ];
	      newP = pConc + delP;
	      if ( newP <= 0.0 ) {
		solution[ pNode->pEqn ] = guessNewConc( pConc, delP );
	      }
	      else {
		solution[ pNode->pEqn ] = newP;
	      }
	    }
	  }
	}
      }
    }
  }
}

/* functions to update the solution for the full-LU and
   modified two-level Newton methods
   */

void NUMD2update(TWOdevice *pDevice, double delV, BOOLEAN updateBoundary)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pContact = pDevice->pLastContact;
  double delPsi, delN, delP, *incVpn, *solution = pDevice->dcSolution;
  
  delV = -delV / VNorm;
  if ( updateBoundary ) {
    /* update the boundary condition for the last contact */
    numContactNodes = pContact->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pContact->pNodes[ index ];
      pNode->psi += delV;
    }
  }
  
  /* the equations have been solved while computing the conductances */
  /* solution is in dcDeltaSolution, so use it */

  incVpn = pDevice->dcDeltaSolution;
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) {
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  delPsi = incVpn[ pNode->psiEqn ] * delV;
	  solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	    delN = incVpn[ pNode->nEqn ] * delV;
	    solution[ pNode->nEqn ] = pNode->nConc + delN;
	  }
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	    delP = incVpn[ pNode->pEqn ] * delV;
	    solution[ pNode->pEqn ] = pNode->pConc + delP;
	  }
	}
      }
    }
  }
}


void NBJT2update(TWOdevice *pDevice, double delVce, double delVbe, 
                 BOOLEAN updateBoundary)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pColContact = pDevice->pFirstContact;
  TWOcontact *pBaseContact = pDevice->pFirstContact->next;
  double delPsi, delN, delP, *incVce, *incVbe, *solution = pDevice->dcSolution;
  
  /* normalize the voltages for calculations */
  
  if ( delVce != 0.0 ) {
    delVce = delVce / VNorm;
    if ( updateBoundary ) {
      numContactNodes = pColContact->numNodes;
      for ( index = 0; index < numContactNodes; index++ ) {
	pNode = pColContact->pNodes[ index ];
	pNode->psi += delVce;
      }
    }
  }
  if ( delVbe != 0.0 ) {
    delVbe = delVbe / VNorm;
    if ( updateBoundary ) {
      numContactNodes = pBaseContact->numNodes;
      for ( index = 0; index < numContactNodes; index++ ) {
	pNode = pBaseContact->pNodes[ index ];
	pNode->psi += delVbe;
      }
    }
  }
  
  /* use solution from computeConductance to do update */
  /* set incVce = dcDeltaSolution; incVbe = copiedSolution */
  
  incVce = pDevice->dcDeltaSolution;
  incVbe = pDevice->copiedSolution;
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) {
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  delPsi = (incVce[ pNode->psiEqn ] * delVce
		    + incVbe[ pNode->psiEqn ] * delVbe);
	  solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	    delN = (incVce[ pNode->nEqn ] * delVce
		    + incVbe[ pNode->nEqn ] * delVbe);
	    solution[ pNode->nEqn ] = pNode->nConc + delN;
	  }
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	    delP = (incVce[ pNode->pEqn ] * delVce
		    + incVbe[ pNode->pEqn ] * delVbe);
	    solution[ pNode->pEqn ] = pNode->pConc + delP;
	  }
	}
      }
    }
  }
}


void NUMOSupdate(TWOdevice *pDevice, double delVdb, double delVsb, 
                 double delVgb, BOOLEAN updateBoundary)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex, numContactNodes;
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
  double delPsi, delN, delP;
  double *incVdb, *incVsb, *incVgb, *solution = pDevice->dcSolution;
  
  /* normalize the voltages for calculations */
  if ( delVdb != 0.0 ) {
    delVdb = delVdb / VNorm;
    if ( updateBoundary ) {
      numContactNodes = pDContact->numNodes;
      for ( index = 0; index < numContactNodes; index++ ) {
	pNode = pDContact->pNodes[ index ];
	pNode->psi += delVdb;
      }
    }
  }
  if ( delVsb != 0.0 ) {
    delVsb = delVsb / VNorm;
    if ( updateBoundary ) {
      numContactNodes = pSContact->numNodes;
      for ( index = 0; index < numContactNodes; index++ ) {
	pNode = pSContact->pNodes[ index ];
	pNode->psi += delVsb;
      }
    }
  }
  if ( delVgb != 0.0 ) {
    delVgb = delVgb / VNorm;
    if ( updateBoundary ) {
      numContactNodes = pGContact->numNodes;
      for ( index = 0; index < numContactNodes; index++ ) {
	pNode = pGContact->pNodes[ index ];
	pNode->psi += delVgb;
      }
    }
  }
  
  /* use solution from computeConductance to do update */
  
  incVdb = pDevice->dcDeltaSolution;
  incVsb = pDevice->copiedSolution;
  incVgb = pDevice->rhsImag;
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) {
	pNode = pElem->pNodes[ index ];
	if ( pNode->nodeType != CONTACT ) {
	  delPsi = (incVdb[ pNode->psiEqn ] * delVdb
		    + incVsb[ pNode->psiEqn ] * delVsb
		    + incVgb[ pNode->psiEqn ] * delVgb);
	  solution[ pNode->psiEqn ] = pNode->psi + delPsi;
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == N_TYPE)) ) {
	    delN = (incVdb[ pNode->nEqn ] * delVdb
		    + incVsb[ pNode->nEqn ] * delVsb
		    + incVgb[ pNode->nEqn ] * delVgb);
	    solution[ pNode->nEqn ] = pNode->nConc + delN;
	  }
	  if ( pElem->elemType == SEMICON
	      && (!OneCarrier || (OneCarrier == P_TYPE)) ) {
	    delP = (incVdb[ pNode->pEqn ] * delVdb
		    + incVsb[ pNode->pEqn ] * delVsb
		    + incVgb[ pNode->pEqn ] * delVgb);
	    solution[ pNode->pEqn ] = pNode->pConc + delP;
	  }
	}
      }
    }
  }
}


void storeNewRhs(TWOdevice *pDevice, TWOcontact *pContact)
{
  int index, i, numContactNodes;
  TWOelem *pElem;
  TWOnode *pNode, *pHNode = NULL, *pVNode = NULL;
  TWOedge *pHEdge = NULL, *pVEdge = NULL;
  double *rhs = pDevice->rhs;
  
  /* zero the rhs before loading in the new rhs */
  for ( index = 1; index <= pDevice->numEqns ; index++ ) {
    rhs[ index ] = 0.0;
  }
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL) {
	/* found an element to which this node belongs */
	switch ( i ) {
	case 0:
	  /* the TL element of this node */
	  pHNode = pElem->pBLNode;
	  pVNode = pElem->pTRNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  break;
	case 1:
	  /* the TR element */
	  pHNode = pElem->pBRNode;
	  pVNode = pElem->pTLNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  break;
	case 2:
	  /* the BR element */
	  pHNode = pElem->pTRNode;
	  pVNode = pElem->pBLNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  break;
	case 3:
	  /* the BL element */
	  pHNode = pElem->pTLNode;
	  pVNode = pElem->pBRNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  break;
	default:
	  printf( "storeNewRhs: shouldn't be here\n");
	  break;
	}
	if ( pHNode->nodeType != CONTACT ) {
	  /* contribution to the x nodes */
	  rhs[ pHNode->psiEqn ] += pElem->epsRel * 0.5 * pElem->dyOverDx;
	  if ( pElem->elemType == SEMICON ) {
	    if ( !OneCarrier ) {
	      rhs[ pHNode->nEqn ] -= 0.5 * pElem->dy * pHEdge->dJnDpsiP1;
	      rhs[ pHNode->pEqn ] -= 0.5 * pElem->dy * pHEdge->dJpDpsiP1;
	    } else if ( OneCarrier == N_TYPE ) {
	      rhs[ pHNode->nEqn ] -= 0.5 * pElem->dy * pHEdge->dJnDpsiP1;
	    } else if ( OneCarrier == P_TYPE ) {
	      rhs[ pHNode->pEqn ] -= 0.5 * pElem->dy * pHEdge->dJpDpsiP1;
	    }
	  }
	}
	if ( pVNode->nodeType != CONTACT ) {
	  /* contribution to the y nodes */
	  rhs[ pVNode->psiEqn ] += pElem->epsRel * 0.5 * pElem->dxOverDy;
	  if ( pElem->elemType == SEMICON ) {
	    if ( !OneCarrier ) {
	      rhs[ pVNode->nEqn ] -= 0.5 * pElem->dx * pVEdge->dJnDpsiP1;
	      rhs[ pVNode->pEqn ] -= 0.5 * pElem->dx * pVEdge->dJpDpsiP1;
	    } else if ( OneCarrier == N_TYPE ) {
	      rhs[ pVNode->nEqn ] -= 0.5 * pElem->dx * pVEdge->dJnDpsiP1;
	    } else if ( OneCarrier == P_TYPE ) {
	      rhs[ pVNode->pEqn ] -= 0.5 * pElem->dx * pVEdge->dJpDpsiP1;
	    }
	  }
	}
      }
    }
  }
}
