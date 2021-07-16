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
 * setup matrix pointers to Jacobian values and 
 * store direct pointers with the nodes
 */

void 
  TWOPjacBuild(TWOdevice *pDevice)
{
  SMPmatrix *matrix = pDevice->matrix;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOchannel *pCh;
  int eIndex, nIndex;
  int nextIndex;			/* index of node to find next element */
  int psiEqn, pEqn;			/* scratch for deref'd eqn numbers */
  int psiEqnTL = 0, pEqnTL = 0;
  int psiEqnTR = 0, pEqnTR = 0;
  int psiEqnBR = 0, pEqnBR = 0;
  int psiEqnBL = 0, pEqnBL = 0;
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
	pEqn = pNode->pEqn;
	pNode->nEqn = 0;
	/* pointers for additional terms */
	pNode->fPsiP = spGetElement( matrix, psiEqn, pEqn );
	pNode->fPPsi = spGetElement( matrix, pEqn, psiEqn );
	pNode->fPP = spGetElement( matrix, pEqn, pEqn );
      } else {
	pEqn = 0;
      }
      /* save equation indices */
      switch ( nIndex ) {
      case 0: /* TL Node */
	psiEqnTL = psiEqn;
	pEqnTL = pEqn;
	break;
      case 1: /* TR Node */
	psiEqnTR = psiEqn;
	pEqnTR = pEqn;
	break;
      case 2: /* BR Node */
	psiEqnBR = psiEqn;
	pEqnBR = pEqn;
	break;
      case 3: /* BL Node */
	psiEqnBL = psiEqn;
	pEqnBL = pEqn;
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
      pNode->fPPsiiP1    = spGetElement( matrix, pEqnTL, psiEqnTR );
      pNode->fPPiP1      = spGetElement( matrix, pEqnTL, pEqnTR );
      pNode->fPPsijP1    = spGetElement( matrix, pEqnTL, psiEqnBL );
      pNode->fPPjP1      = spGetElement( matrix, pEqnTL, pEqnBL );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fPPsiiP1jP1 = spGetElement( matrix, pEqnTL, psiEqnBR );
	pNode->fPPiP1jP1   = spGetElement( matrix, pEqnTL, pEqnBR );
      }
    }
    
    pNode = pElem->pTRNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, psiEqnTR, psiEqnTL );
    pNode->fPsiPsijP1 = spGetElement(matrix, psiEqnTR, psiEqnBR );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fPPsiiM1    = spGetElement( matrix, pEqnTR, psiEqnTL );
      pNode->fPPiM1      = spGetElement( matrix, pEqnTR, pEqnTL );
      pNode->fPPsijP1    = spGetElement( matrix, pEqnTR, psiEqnBR );
      pNode->fPPjP1      = spGetElement( matrix, pEqnTR, pEqnBR );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fPPsiiM1jP1 = spGetElement( matrix, pEqnTR, psiEqnBL );
	pNode->fPPiM1jP1   = spGetElement( matrix, pEqnTR, pEqnBL );
      }
    }
    
    pNode = pElem->pBRNode;
    pNode->fPsiPsiiM1 = spGetElement(matrix, psiEqnBR, psiEqnBL );
    pNode->fPsiPsijM1 = spGetElement(matrix, psiEqnBR, psiEqnTR );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fPPsiiM1    = spGetElement( matrix, pEqnBR, psiEqnBL );
      pNode->fPPiM1      = spGetElement( matrix, pEqnBR, pEqnBL );
      pNode->fPPsijM1    = spGetElement( matrix, pEqnBR, psiEqnTR );
      pNode->fPPjM1      = spGetElement( matrix, pEqnBR, pEqnTR );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fPPsiiM1jM1 = spGetElement( matrix, pEqnBR, psiEqnTL );
	pNode->fPPiM1jM1   = spGetElement( matrix, pEqnBR, pEqnTL );
      }
    }
    
    pNode = pElem->pBLNode;
    pNode->fPsiPsiiP1 = spGetElement(matrix, psiEqnBL, psiEqnBR );
    pNode->fPsiPsijM1 = spGetElement(matrix, psiEqnBL, psiEqnTL );
    if ( pElem->elemType == SEMICON ) {
      /* continuity equation pointers */
      pNode->fPPsiiP1    = spGetElement( matrix, pEqnBL, psiEqnBR );
      pNode->fPPiP1      = spGetElement( matrix, pEqnBL, pEqnBR );
      pNode->fPPsijM1    = spGetElement( matrix, pEqnBL, psiEqnTL );
      pNode->fPPjM1      = spGetElement( matrix, pEqnBL, pEqnTL );
      /* Surface Mobility Model depends on diagonal node values */
      if ( MobDeriv && SurfaceMobility && pElem->channel ) {
	pNode->fPPsiiP1jM1 = spGetElement( matrix, pEqnBL, psiEqnTR );
	pNode->fPPiP1jM1   = spGetElement( matrix, pEqnBL, pEqnTR );
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
	  pEqn = pNode->pEqn;
	  if ( pCh->type % 2 == 0 ) { /* Vertical Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left Side */
	      pNode->fPPsiIn   = spGetElement( matrix, pEqn, psiEqnInM );
	      pNode->fPPsiInP1 = spGetElement( matrix, pEqn, psiEqnInP );
	      pNode->fPPsiOx   = spGetElement( matrix, pEqn, psiEqnOxM );
	      pNode->fPPsiOxP1 = spGetElement( matrix, pEqn, psiEqnOxP );
	    } else { /* Right Side */
	      pNode->fPPsiInM1 = spGetElement( matrix, pEqn, psiEqnInM );
	      pNode->fPPsiIn   = spGetElement( matrix, pEqn, psiEqnInP );
	      pNode->fPPsiOxM1 = spGetElement( matrix, pEqn, psiEqnOxM );
	      pNode->fPPsiOx   = spGetElement( matrix, pEqn, psiEqnOxP );
	    }
	  } else { /* Horizontal Slice */
	    if ( nIndex == 0 || nIndex == 3 ) { /* Left (Top?) Side : bug 483 */
	      pNode->fPPsiIn   = spGetElement( matrix, pEqn, psiEqnInM );
	      pNode->fPPsiInP1 = spGetElement( matrix, pEqn, psiEqnInP );
	      pNode->fPPsiOx   = spGetElement( matrix, pEqn, psiEqnOxM );
	      pNode->fPPsiOxP1 = spGetElement( matrix, pEqn, psiEqnOxP );
	    } else { /* Bottom Side */
	      pNode->fPPsiInM1 = spGetElement( matrix, pEqn, psiEqnInM );
	      pNode->fPPsiIn   = spGetElement( matrix, pEqn, psiEqnInP );
	      pNode->fPPsiOxM1 = spGetElement( matrix, pEqn, psiEqnOxM );
	      pNode->fPPsiOx   = spGetElement( matrix, pEqn, psiEqnOxP );
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
  TWOPsysLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
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
  double rhsP;
  double nConc, pConc;
  double perTime = 0.0;
  
  /* first compute the currents and derivatives */
  TWOPcommonTerms( pDevice, FALSE, tranAnalysis, info );
  
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
	*(pNode->fPsiPsi) += dyOverDx + dxOverDy;
	if ( pElem->elemType == SEMICON ) {
	  nConc = pDevice->devState0 [pNode->nodeN];
	  pConc = pDevice->devState0 [pNode->nodeP];

	  *(pNode->fPsiPsi) += dxdy * nConc;
	  *(pNode->fPsiP) -= dxdy;
	  *(pNode->fPPsi) -= dy * pHEdge->dJpDpsiP1 + dx * pVEdge->dJpDpsiP1;
	  pRhs[ pNode->psiEqn ] += dxdy * (pNode->netConc + pConc - nConc);
	  
	  /* Handle generation terms */
	  *(pNode->fPP) += dxdy * pNode->dUdP;
	  *(pNode->fPPsi) += dxdy * pNode->dUdN * nConc;
	  rhsP =   dxdy * pNode->uNet;
	  pRhs[ pNode->pEqn ] -= rhsP;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
	    *(pNode->fPP) += dxdy * perTime;
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
	pRhs[ pNode->pEqn ] -= dy * pTEdge->jp + dx * pLEdge->jp;
	*(pNode->fPP) += dy * pTEdge->dJpDp + dx * pLEdge->dJpDp;
	*(pNode->fPPsiiP1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pTEdge->dJpDpP1;
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
	pRhs[ pNode->pEqn ] -= -dy * pTEdge->jp + dx * pREdge->jp;
	*(pNode->fPP) += -dy * pTEdge->dJpDpP1 + dx * pREdge->dJpDp;
	*(pNode->fPPsiiM1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pTEdge->dJpDp;
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
	pRhs[ pNode->pEqn ] -= -dy * pBEdge->jp - dx * pREdge->jp;
	*(pNode->fPP) += -dy * pBEdge->dJpDpP1 - dx * pREdge->dJpDpP1;
	*(pNode->fPPsiiM1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pBEdge->dJpDp;
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
	pRhs[ pNode->pEqn ] -= dy * pBEdge->jp - dx * pLEdge->jp;
	*(pNode->fPP) += dy * pBEdge->dJpDp - dx * pLEdge->dJpDpP1;
	*(pNode->fPPsiiP1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pBEdge->dJpDpP1;
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
	TWOPmobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}


/* this function used only for direct method ac analysis 
   Used to load only the dc Jacobian matrix. Rhs is unaffected
   */

void 
  TWOPjacLoad(TWOdevice *pDevice)
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
  double nConc;
  
  /* first compute the currents and derivatives */
  TWOPcommonTerms( pDevice, FALSE, FALSE, NULL );
  
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
	  nConc = pDevice->devState0 [pNode->nodeN];
	  *(pNode->fPsiPsi) += dxdy * nConc;
	  *(pNode->fPsiP) -= dxdy;
	  *(pNode->fPPsi) -= dy * pHEdge->dJpDpsiP1 + dx * pVEdge->dJpDpsiP1;
	  
	  /* Handle generation terms */
	  *(pNode->fPP) += dxdy * pNode->dUdP;
	  *(pNode->fPPsi) += dxdy * pNode->dUdN * nConc;
	}
      }
    }
    
    /* Handle neighbor and edge dependent terms */
    pNode = pElem->pTLNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fPP) += dy * pTEdge->dJpDp + dx * pLEdge->dJpDp;
	*(pNode->fPPsiiP1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pTEdge->dJpDpP1;
	*(pNode->fPPsijP1) += dx * pLEdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pLEdge->dJpDpP1;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijP1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fPP) += -dy * pTEdge->dJpDpP1 + dx * pREdge->dJpDp;
	*(pNode->fPPsiiM1) += dy * pTEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pTEdge->dJpDp;
	*(pNode->fPPsijP1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjP1) += dx * pREdge->dJpDpP1;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiM1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fPP) += -dy * pBEdge->dJpDpP1 - dx * pREdge->dJpDpP1;
	*(pNode->fPPsiiM1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiM1) -= dy * pBEdge->dJpDp;
	*(pNode->fPPsijM1) += dx * pREdge->dJpDpsiP1;
	*(pNode->fPPjM1) -= dx * pREdge->dJpDp;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      *(pNode->fPsiPsiiP1) -= dyOverDx;
      *(pNode->fPsiPsijM1) -= dxOverDy;
      if ( pElem->elemType == SEMICON ) {
	*(pNode->fPP) += dy * pBEdge->dJpDp - dx * pLEdge->dJpDpP1;
	*(pNode->fPPsiiP1) += dy * pBEdge->dJpDpsiP1;
	*(pNode->fPPiP1) += dy * pBEdge->dJpDpP1;
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
	TWOPmobDeriv( pElem, pCh->type, ds );
	pElem = pElem->pElems[ nextIndex ];
      }
    } /* endfor pCh != NULL */
  } /* endif MobDeriv and SurfaceMobility */
}

/* load only the Rhs vector */
void 
  TWOPrhsLoad(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pHEdge, *pVEdge;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  int index, eIndex;
  double *pRhs = pDevice->rhs;
  double dx, dy, dxdy, dyOverDx, dxOverDy;
  double dPsiT, dPsiB, dPsiL, dPsiR;
  double rhsP;
  double nConc, pConc;
  double perTime;
  
  /* first compute the currents */
  TWOPcommonTerms( pDevice, TRUE, tranAnalysis, info );
  
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
	  rhsP =   dxdy * pNode->uNet;
	  pRhs[ pNode->pEqn ] -= rhsP;
	  
	  /* Handle dXdT continuity terms */
	  if ( tranAnalysis ) {
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
	pRhs[ pNode->pEqn ] -= dy * pTEdge->jp + dx * pLEdge->jp;
      }
    }
    pNode = pElem->pTRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiT - dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->pEqn ] -= -dy * pTEdge->jp + dx * pREdge->jp;
      }
    }
    pNode = pElem->pBRNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= dyOverDx * dPsiB + dxOverDy * dPsiR;
      if ( pElem->elemType == SEMICON ) {
	pRhs[ pNode->pEqn ] -= -dy * pBEdge->jp - dx * pREdge->jp;
      }
    }
    pNode = pElem->pBLNode;
    if ( pNode->nodeType != CONTACT ) {
      pRhs[ pNode->psiEqn ] -= -dyOverDx * dPsiB + dxOverDy * dPsiL;
      if ( pElem->elemType == SEMICON ) {
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
  TWOPcommonTerms(TWOdevice *pDevice, BOOLEAN currentOnly, 
                  BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOelem *pElem;
  TWOedge *pEdge;
  TWOnode *pNode;
  int index, eIndex;
  int nextIndex;			/* index of node to find next element */
  double psi1, psi2, refPsi, pC, pP1;
  double dPsiP;
  double bPsiP, dbPsiP, bMPsiP, dbMPsiP;
  double muP, dMuP, rDx, rDy;
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
	    nConc = pNode->nie * exp( psi - refPsi );
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
	  if ( tranAnalysis && pNode->nodeType != CONTACT ) {
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
	  dPsiP = pEdge->dPsi - pEdge->dVBand;
	  bernoulli( dPsiP, &bPsiP, &dbPsiP,
		    &bMPsiP, &dbMPsiP, !currentOnly);
	  if ( index <= 1 ) {
	    pC = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeP);
	    pP1 = *(pDevice->devState0 + pElem->pNodes[ index+1 ]->nodeP);
	  } else {
	    pC = *(pDevice->devState0 + pElem->pNodes[(index+1)%4]->nodeP);
	    pP1 = *(pDevice->devState0 + pElem->pNodes[ index ]->nodeP);
	  }
	  pEdge->wdfp = bPsiP * pC - bMPsiP * pP1;
	  pEdge->jp = 0.0;
	  if ( !currentOnly ) {
	    pEdge->dWpDpsiP1 = dbPsiP * pC - dbMPsiP * pP1;
	    pEdge->dWpDp     = bPsiP;
	    pEdge->dWpDpP1   = - bMPsiP;
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
	TWOPmobility( pElem, eSurf );
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
	  muP = pElem->mup0;
	  dMuP = 0.0;
	  dPsiP = pEdge->dPsi - pEdge->dVBand;
	  if ( index%2 == 0 ) {
	    MOBfieldDep( pElem->matlInfo, HOLE, - dPsiP * rDx, &muP, &dMuP );
	  } else {
	    MOBfieldDep( pElem->matlInfo, HOLE, - dPsiP * rDy, &muP, &dMuP );
	  }
	} else {
	  /* Retrieve previously calculated value. */
	  muP = pElem->mup;
	  dMuP = 0.0;
	}
	switch ( index ) {
	case 0:
	  muP *= pEdge->kPos * rDx;
	  dMuP *= pEdge->kPos * rDx * rDx;
	  break;
	case 1:
	  muP *= pEdge->kNeg * rDy;
	  dMuP *= pEdge->kNeg * rDy * rDy;
	  break;
	case 2:
	  muP *= pEdge->kNeg * rDx;
	  dMuP *= pEdge->kNeg * rDx * rDx;
	  break;
	case 3:
	  muP *= pEdge->kPos * rDy;
	  dMuP *= pEdge->kPos * rDy * rDy;
	  break;
	}
	/* Now that the mobility for this edge is known, do current */
	pEdge->jp += muP * pEdge->wdfp;
	if ( !currentOnly ) {
	  pEdge->dJpDpsiP1 += muP * pEdge->dWpDpsiP1;
	  pEdge->dJpDp += muP * pEdge->dWpDp;
	  pEdge->dJpDpP1 += muP * pEdge->dWpDpP1;
	  if ( MobDeriv && (!pElem->channel) ) {
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
