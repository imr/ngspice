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
      pNode->fPsiPsi = spGetElement( matrix, psiEqn, psiEqn );
      
      if ( pElem->elemType == SEMICON ) {
	/* get continuity-coupling terms */
	nEqn = pNode->nEqn;
	pNode->pEqn = 0;		/* Throw pEqn number into garbage. */
	/* pointers for additional terms */
	pNode->fPsiN = spGetElement( matrix, psiEqn, nEqn );
	pNode->fNPsi = spGetElement( matrix, nEqn, psiEqn );
	pNode->fNN = spGetElement( matrix, nEqn, nEqn );
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
    pNode->fPsiPsiiP1 = spGetElement(matrix, psiEqnTL, psiEqnTR );
    pNode->fPsiPsijP1 = spGetElement(matrix, psiEqnTL, psiEqnBL );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fNPsiiP1    = spGetElement( matrix, nEqnTL, psiEqnTR );
      pNode->fNNiP1      = spGetElement( matrix, nEqnTL, nEqnTR );
      pNode->fNPsijP1    = spGetElement( matrix, nEqnTL, psiEqnBL );
      pNode->fNNjP1      = spGetElement( matrix, nEqnTL, nEqnBL );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fNPsiiP1jP1 = spGetElement( matrix, nEqnTL, psiEqnBR );
	pNode->fNNiP1jP1   = spGetElement( matrix, nEqnTL, nEqnBR );
      }
    }
    
    pNode = pElem->pTRNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, psiEqnTR, psiEqnTL );
    pNode->fPsiPsijP1 = spGetElement(matrix, psiEqnTR, psiEqnBR );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fNPsiiM1    = spGetElement( matrix, nEqnTR, psiEqnTL );
      pNode->fNNiM1      = spGetElement( matrix, nEqnTR, nEqnTL );
      pNode->fNPsijP1    = spGetElement( matrix, nEqnTR, psiEqnBR );
      pNode->fNNjP1      = spGetElement( matrix, nEqnTR, nEqnBR );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fNPsiiM1jP1 = spGetElement( matrix, nEqnTR, psiEqnBL );
	pNode->fNNiM1jP1   = spGetElement( matrix, nEqnTR, nEqnBL );
      }
    }
    
    pNode = pElem->pBRNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, psiEqnBR, psiEqnBL );
    pNode->fPsiPsijM1 = spGetElement(matrix, psiEqnBR, psiEqnTR );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fNPsiiM1    = spGetElement( matrix, nEqnBR, psiEqnBL );
      pNode->fNNiM1      = spGetElement( matrix, nEqnBR, nEqnBL );
      pNode->fNPsijM1    = spGetElement( matrix, nEqnBR, psiEqnTR );
      pNode->fNNjM1      = spGetElement( matrix, nEqnBR, nEqnTR );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fNPsiiM1jM1 = spGetElement( matrix, nEqnBR, psiEqnTL );
	pNode->fNNiM1jM1   = spGetElement( matrix, nEqnBR, nEqnTL );
      }
    }
    
    pNode = pElem->pBLNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, psiEqnBL, psiEqnBR );
    pNode->fPsiPsijM1 = spGetElement(matrix, psiEqnBL, psiEqnTL );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fNPsiiP1    = spGetElement( matrix, nEqnBL, psiEqnBR );
      pNode->fNNiP1      = spGetElement( matrix, nEqnBL, nEqnBR );
      pNode->fNPsijM1    = spGetElement( matrix, nEqnBL, psiEqnTL );
      pNode->fNNjM1      = spGetElement( matrix, nEqnBL, nEqnTL );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fNPsiiP1jM1 = spGetElement( matrix, nEqnBL, psiEqnTR );
	pNode->fNNiP1jM1   = spGetElement( matrix, nEqnBL, nEqnTR );
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
	      pNode->fNPsiIn   = spGetElement( matrix, nEqn, psiEqnInM );
	      pNode->fNPsiInP1 = spGetElement( matrix, nEqn, psiEqnInP );
	      pNode->fNPsiOx   = spGetElement( matrix, nEqn, psiEqnOxM );
	      pNode->fNPsiOxP1 = spGetElement( matrix, nEqn, psiEqnOxP );
	    } else { /* Right Side */
	      pNode->fNPsiInM1 = spGetElement( matrix, nEqn, psiEqnInM );
	      pNode->fNPsiIn   = spGetElement( matrix, nEqn, psiEqnInP );
	      pNode->fNPsiOxM1 = spGetElement( matrix, nEqn, psiEqnOxM );
	      pNode->fNPsiOx   = spGetElement( matrix, nEqn, psiEqnOxP );
	    }
	  } else { /* Horizontal Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left (Top?) Side : bug 483 */
	      pNode->fNPsiIn   = spGetElement( matrix, nEqn, psiEqnInM );
	      pNode->fNPsiInP1 = spGetElement( matrix, nEqn, psiEqnInP );
	      pNode->fNPsiOx   = spGetElement( matrix, nEqn, psiEqnOxM );
	      pNode->fNPsiOxP1 = spGetElement( matrix, nEqn, psiEqnOxP );
	    } else { /* Bottom Side */
	      pNode->fNPsiInM1 = spGetElement( matrix, nEqn, psiEqnInM );
	      pNode->fNPsiIn   = spGetElement( matrix, nEqn, psiEqnInP );
	      pNode->fNPsiOxM1 = spGetElement( matrix, nEqn, psiEqnOxM );
	      pNode->fNPsiOx   = spGetElement( matrix, nEqn, psiEqnOxP );
	    }
	  }
	} /* endfor nIndex */
	pElem = pElem->pElems[ nextIndex ];
      } /* endwhile pElem */
    } /* endfor pCh */
  } /* endif SurfaceMobility */
}


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
  spClear( pDevice->matrix );
  
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
  spClear( pDevice->matrix );
  
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
