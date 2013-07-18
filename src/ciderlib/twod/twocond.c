/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/* Functions to compute terminal conductances & currents. */

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/bool.h"
#include "ngspice/spmatrix.h"
#include "twoddefs.h"
#include "twodext.h"


void 
   NUMD2conductance(TWOdevice *pDevice, BOOLEAN tranAnalysis, 
                    double *intCoeff, double *gd)
{
  TWOcontact *pContact = pDevice->pFirstContact;
  double *incVpn;
  BOOLEAN deltaVContact = FALSE;
  
  /* 
   * store the new rhs for computing the incremental quantities
   * with the second to last node. solve the system of equations
   */
  incVpn = pDevice->dcDeltaSolution;
  storeNewRhs( pDevice, pDevice->pLastContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVpn, NULL, NULL);
  
  incVpn = pDevice->dcDeltaSolution;
  *gd = contactConductance( pDevice, pContact, deltaVContact, incVpn,
			  tranAnalysis, intCoeff );
  *gd *= - GNorm * pDevice->width * LNorm;
}

void 
   NBJT2conductance(TWOdevice *pDevice, BOOLEAN tranAnalysis, 
                    double *intCoeff, double *dIeDVce, double *dIcDVce, 
	   	    double *dIeDVbe, double *dIcDVbe)
{
  TWOcontact *pEmitContact = pDevice->pLastContact;
  TWOcontact *pColContact = pDevice->pFirstContact;
  TWOcontact *pBaseContact = pDevice->pFirstContact->next;
  double width = pDevice->width;
  double *incVce, *incVbe;
  
  /* 
   * store the new rhs for computing the incremental quantities
   * incVce (dcDeltaSolution) and incVbe (copiedSolution) are used to
   * store the incremental quantities associated with Vce and Vbe
   */
  
  incVce = pDevice->dcDeltaSolution;
  incVbe = pDevice->copiedSolution;
  storeNewRhs( pDevice, pColContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVce, NULL, NULL);
  storeNewRhs( pDevice, pBaseContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVbe, NULL, NULL);
  
  *dIeDVce = contactConductance( pDevice, pEmitContact, FALSE, incVce,
				tranAnalysis, intCoeff );
  *dIeDVbe = contactConductance( pDevice, pEmitContact, FALSE, incVbe,
				tranAnalysis, intCoeff );
  
  *dIcDVce = contactConductance( pDevice, pColContact, TRUE, incVce,
				tranAnalysis, intCoeff );
  *dIcDVbe = contactConductance( pDevice, pColContact, FALSE, incVbe,
				tranAnalysis, intCoeff );
  *dIeDVce *= GNorm * width * LNorm;
  *dIcDVce *= GNorm * width * LNorm;
  *dIeDVbe *= GNorm * width * LNorm;
  *dIcDVbe *= GNorm * width * LNorm;
}

void 
   NUMOSconductance(TWOdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff, 
                    struct mosConductances *dIdV)
{
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
  double width = pDevice->width;
  double *incVdb, *incVsb, *incVgb;
  
  /* 
   * store the new rhs for computing the incremental quantities
   * incVdb (dcDeltaSolution)
   */
  
  incVdb = pDevice->dcDeltaSolution;
  incVsb = pDevice->copiedSolution;
  incVgb = pDevice->rhsImag;
  storeNewRhs( pDevice, pDContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVdb, NULL, NULL);
  storeNewRhs( pDevice, pSContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVsb, NULL, NULL);
  storeNewRhs( pDevice, pGContact );
  spSolve( pDevice->matrix, pDevice->rhs, incVgb, NULL, NULL);
  
  dIdV->dIdDVdb = contactConductance( pDevice, pDContact, TRUE,
				     incVdb, tranAnalysis, intCoeff );
  dIdV->dIsDVdb = contactConductance( pDevice, pSContact, FALSE,
				     incVdb, tranAnalysis, intCoeff );
  dIdV->dIgDVdb = GateTypeConductance( pDevice, pGContact, FALSE,
				      incVdb, tranAnalysis, intCoeff );
  dIdV->dIdDVsb = contactConductance( pDevice, pDContact, FALSE,
				     incVsb, tranAnalysis, intCoeff );
  dIdV->dIsDVsb = contactConductance( pDevice, pSContact, TRUE,
				     incVsb, tranAnalysis, intCoeff );
  dIdV->dIgDVsb = GateTypeConductance( pDevice, pGContact, FALSE,
				      incVsb, tranAnalysis, intCoeff );
  dIdV->dIdDVgb = contactConductance( pDevice, pDContact, FALSE,
				     incVgb, tranAnalysis, intCoeff );
  dIdV->dIsDVgb = contactConductance( pDevice, pSContact, FALSE,
				     incVgb, tranAnalysis, intCoeff );
  dIdV->dIgDVgb = GateTypeConductance( pDevice, pGContact, TRUE,
				      incVgb, tranAnalysis, intCoeff );
  
  dIdV->dIdDVdb *= GNorm * width * LNorm;
  dIdV->dIdDVsb *= GNorm * width * LNorm;
  dIdV->dIdDVgb *= GNorm * width * LNorm;
  dIdV->dIsDVdb *= GNorm * width * LNorm;
  dIdV->dIsDVsb *= GNorm * width * LNorm;
  dIdV->dIsDVgb *= GNorm * width * LNorm;
  dIdV->dIgDVdb *= GNorm * width * LNorm;
  dIdV->dIgDVsb *= GNorm * width * LNorm;
  dIdV->dIgDVgb *= GNorm * width * LNorm;
  
}

double
  contactCurrent(TWOdevice *pDevice, TWOcontact *pContact)
{
  /* computes the current through the contact given in pContact */
  int index, i, numContactNodes;
  TWOnode *pNode;
  TWOelem *pElem;
  TWOedge *pHEdge, *pVEdge;
  double dx, dy;
  double jTotal = 0.0;
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL ) {
	dx = 0.5 * pElem->dx;
	dy = 0.5 * pElem->dy;
	switch ( i ) {
	case 0:
	  /* Bottom Right node */
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  jTotal += pElem->epsRel * ( -dy * pHEdge->jd - dx * pVEdge->jd );
	  if ( pElem->elemType == SEMICON ) {
	    jTotal += -dy * (pHEdge->jn + pHEdge->jp)
	      -dx * (pVEdge->jn + pVEdge->jp);
	  }
	  break;
	case 1:
	  /* Bottom Left node */
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  jTotal += pElem->epsRel * ( dy * pHEdge->jd - dx * pVEdge->jd );
	  if ( pElem->elemType == SEMICON ) {
	    jTotal += dy * (pHEdge->jn + pHEdge->jp)
	      -dx * (pVEdge->jn + pVEdge->jp);
	  }
	  break;
	case 2:
	  /* Top Left node */
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  jTotal += pElem->epsRel * ( dy * pHEdge->jd + dx * pVEdge->jd );
	  if ( pElem->elemType == SEMICON ) {
	    jTotal += dy * (pHEdge->jn + pHEdge->jp)
	      + dx * (pVEdge->jn + pVEdge->jp);
	  }
	  break;
	case 3:
	  /* Top Right Node */
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  jTotal += pElem->epsRel * ( -dy * pHEdge->jd + dx * pVEdge->jd );
	  if ( pElem->elemType == SEMICON ) {
	    jTotal += -dy * (pHEdge->jn + pHEdge->jp)
	      + dx * (pVEdge->jn + pVEdge->jp);
	  }
	  break;
	}
      }
    }
  }
  
  return( jTotal * pDevice->width * LNorm * JNorm );
}

double 
  oxideCurrent(TWOdevice *pDevice, TWOcontact *pContact, 
               BOOLEAN tranAnalysis)
{
  /* computes the current through the contact given in pContact */
  int index, i, numContactNodes;
  TWOnode *pNode;
  TWOelem *pElem;
  TWOedge *pHEdge, *pVEdge;
  double dx, dy;
  double jTotal = 0.0;
  
  if ( !tranAnalysis ) {
    return( jTotal );
  }
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL ) {
	dx = 0.5 * pElem->dx;
	dy = 0.5 * pElem->dy;
	switch ( i ) {
	case 0:
	  /* Bottom Right node */
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  jTotal += pElem->epsRel * ( -dy * pHEdge->jd - dx * pVEdge->jd );
	  break;
	case 1:
	  /* Bottom Left node */
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  jTotal += pElem->epsRel * ( dy * pHEdge->jd - dx * pVEdge->jd );
	  break;
	case 2:
	  /* Top Left node */
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  jTotal += pElem->epsRel * ( dy * pHEdge->jd + dx * pVEdge->jd );
	  break;
	case 3:
	  /* Top Right Node */
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  jTotal += pElem->epsRel * ( -dy * pHEdge->jd + dx * pVEdge->jd );
	  break;
	}
      }
    }
  }
  
  return( jTotal * pDevice->width * LNorm * JNorm );
}


double 
   contactConductance(TWOdevice *pDevice, TWOcontact *pContact, 
                      BOOLEAN delVContact, double *dxDv, 
	              BOOLEAN tranAnalysis, double *intCoeff)
{
  /* computes the conductance of the contact given in pContact */
  int index, i, numContactNodes;
  TWOnode *pNode, *pHNode = NULL, *pVNode = NULL;
  TWOelem *pElem;
  TWOedge *pHEdge = NULL, *pVEdge = NULL;
  double dPsiDv, dnDv, dpDv;
  double gTotal = 0.0;
  int nInc, pInc;
  
  NG_IGNORE(pDevice);

  /* for one carrier the rest of this code relies on appropriate 
     current derivative term to be zero */
  if ( !OneCarrier ) {
    nInc = 1;
    pInc = 2;
  } else {
    nInc = 1;
    pInc = 1;
  } 
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL ) {
	switch ( i ) {
	case 0:
	  /* the TL element */
	  pHNode = pElem->pBLNode;
	  pVNode = pElem->pTRNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  if ( pElem->elemType == SEMICON ) {
	    /* compute the derivatives with n,p */
	    if ( pHNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pHNode->nEqn ]; 
	      dpDv = dxDv[ pHNode->pEqn ];
	      gTotal -= 0.5 * pElem->dy * (pHEdge->dJnDn * dnDv
					   + pHEdge->dJpDp * dpDv);
	    }
	    if ( pVNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pVNode->nEqn ]; 
	      dpDv = dxDv[ pVNode->pEqn ];
	      gTotal -= 0.5 * pElem->dx * (pVEdge->dJnDn * dnDv
					   + pVEdge->dJpDp * dpDv);
	    }
	  }
	  break;
	case 1:
	  /* the TR element */
	  pHNode = pElem->pBRNode;
	  pVNode = pElem->pTLNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  if ( pElem->elemType == SEMICON ) {
	    /* compute the derivatives with n,p */
	    if ( pHNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pHNode->nEqn ]; 
	      dpDv = dxDv[ pHNode->pEqn ];
	      gTotal += 0.5 * pElem->dy * (pHEdge->dJnDnP1 * dnDv
					   + pHEdge->dJpDpP1 * dpDv);
	    }
	    if ( pVNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pVNode->nEqn ]; 
	      dpDv = dxDv[ pVNode->pEqn ];
	      gTotal -= 0.5 * pElem->dx * (pVEdge->dJnDn * dnDv
					   + pVEdge->dJpDp * dpDv);
	    }
	  }
	  break;
	case 2:
	  /* the BR element*/
	  pHNode = pElem->pTRNode;
	  pVNode = pElem->pBLNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  if ( pElem->elemType == SEMICON ) {
	    /* compute the derivatives with n,p */
	    if ( pHNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pHNode->nEqn ]; 
	      dpDv = dxDv[ pHNode->pEqn ];
	      gTotal += 0.5 * pElem->dy * (pHEdge->dJnDnP1 * dnDv
					   + pHEdge->dJpDpP1 * dpDv);
	    }
	    if ( pVNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pVNode->nEqn ]; 
	      dpDv = dxDv[ pVNode->pEqn ];
	      gTotal += 0.5 * pElem->dx * (pVEdge->dJnDnP1 * dnDv
					   + pVEdge->dJpDpP1 * dpDv);
	    }
	  }
	  break;
	case 3:
	  /* the BL element */
	  pHNode = pElem->pTLNode;
	  pVNode = pElem->pBRNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  if ( pElem->elemType == SEMICON ) {
	    /* compute the derivatives with n,p */
	    if ( pHNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pHNode->nEqn ]; 
	      dpDv = dxDv[ pHNode->pEqn ];
	      gTotal -= 0.5 * pElem->dy * (pHEdge->dJnDn * dnDv
					   + pHEdge->dJpDp * dpDv);
	    }
	    if ( pVNode->nodeType != CONTACT ) { 
	      dnDv = dxDv[ pVNode->nEqn ]; 
	      dpDv = dxDv[ pVNode->pEqn ];
	      gTotal += 0.5 * pElem->dx * (pVEdge->dJnDnP1 * dnDv
					   + pVEdge->dJpDpP1 * dpDv);
	    }
	  }
	  break;
	}
	if ( pElem->elemType == SEMICON ) {
	  if ( pHNode->nodeType != CONTACT ) { 
	    dPsiDv = dxDv[ pHNode->psiEqn ];
	    gTotal += 0.5 * pElem->dy * dPsiDv * (pHEdge->dJnDpsiP1 + pHEdge->dJpDpsiP1 ); 
	    if ( delVContact ) {
	      gTotal -= 0.5 * pElem->dy * (pHEdge->dJnDpsiP1 + pHEdge->dJpDpsiP1 );
	    }
	  }
	  if ( pVNode->nodeType != CONTACT ) { 
	    dPsiDv = dxDv[ pVNode->psiEqn ];
	    gTotal += 0.5 * pElem->dx * dPsiDv * (pVEdge->dJnDpsiP1 + pVEdge->dJpDpsiP1 ); 
	    if ( delVContact ) {
	      gTotal -= 0.5 * pElem->dx * (pVEdge->dJnDpsiP1 + pVEdge->dJpDpsiP1 );
	    }
	  }
	}
	if ( tranAnalysis ) {
	  /* add the displacement current terms */
	  if ( pHNode->nodeType != CONTACT ) { 
	    dPsiDv = dxDv[ pHNode->psiEqn ];
	    gTotal -= intCoeff[0] * pElem->epsRel * 0.5 * pElem->dyOverDx * dPsiDv;
	    if ( delVContact ) {
	      gTotal += intCoeff[0] * pElem->epsRel * 0.5 * pElem->dyOverDx;
	    }
	  }
	  if ( pVNode->nodeType != CONTACT ) { 
	    dPsiDv = dxDv[ pVNode->psiEqn ];
	    gTotal -= intCoeff[0] * pElem->epsRel * 0.5 * pElem->dxOverDy * dPsiDv;
	    if ( delVContact ) {
	      gTotal += intCoeff[0] * pElem->epsRel * 0.5 * pElem->dxOverDy;
	    }
	  }
	}
      }
    }
  }
  
  return( gTotal );
}


double 
   oxideConductance(TWOdevice *pDevice, TWOcontact *pContact, 
                    BOOLEAN delVContact, double *dxDv, 
	            BOOLEAN tranAnalysis, double *intCoeff)
{
  /* computes the conductance of the contact given in pContact */
  int index, i, numContactNodes;
  TWOnode *pNode, *pHNode = NULL, *pVNode = NULL;
  TWOelem *pElem;
  double dPsiDv;
  double gTotal = 0.0;
  
  NG_IGNORE(pDevice);

  if ( !tranAnalysis ) {
    return( gTotal );
  }
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL ) {
	switch ( i ) {
	case 0:
	  /* the TL element */
	  pHNode = pElem->pBLNode;
	  pVNode = pElem->pTRNode;
	  break;
	case 1:
	  /* the TR element */
	  pHNode = pElem->pBRNode;
	  pVNode = pElem->pTLNode;
	  break;
	case 2:
	  /* the BR element*/
	  pHNode = pElem->pTRNode;
	  pVNode = pElem->pBLNode;
	  break;
	case 3:
	  /* the BL element */
	  pHNode = pElem->pTLNode;
	  pVNode = pElem->pBRNode;
	  break;
	}
	/* add the displacement current terms */
	if ( pHNode->nodeType != CONTACT ) { 
	  dPsiDv = dxDv[ pHNode->psiEqn ];
	  gTotal -= intCoeff[0] * pElem->epsRel * 0.5 * pElem->dyOverDx * dPsiDv;
	  if ( delVContact ) {
	    gTotal += intCoeff[0] * pElem->epsRel * 0.5 * pElem->dyOverDx;
	  }
	}
	if ( pVNode->nodeType != CONTACT ) { 
	  dPsiDv = dxDv[ pVNode->psiEqn ];
	  gTotal -= intCoeff[0] * pElem->epsRel * 0.5 * pElem->dxOverDy * dPsiDv;
	  if ( delVContact ) {
	    gTotal += intCoeff[0] * pElem->epsRel * 0.5 * pElem->dxOverDy;
	  }
	}
      }
    }
  }
  
  return( gTotal );
}

/* these functions are used for solving the complete system of 
 * equations directly using LU decomposition   1/22/88
 */

void 
   NUMD2current(TWOdevice *pDevice, BOOLEAN tranAnalysis, 
                double *intCoeff, double *id)
{
  TWOcontact *pPContact = pDevice->pFirstContact;
/*  TWOcontact *pNContact = pDevice->pLastContact; */
  double ip, ipPrime, *solution;
/*  double in;*/
  BOOLEAN deltaVContact = FALSE;
  
  solution = pDevice->dcDeltaSolution;
  ip = contactCurrent( pDevice, pPContact );
  /*
  in = contactCurrent( pDevice, pNContact );
  fprintf(stdout, "DIO current: ( %11.4e error )\n", ip+in );
  fprintf(stdout, "     Ip = %11.4e     In = %11.4e\n", ip, in );
  */

  /* 
   * for the additional contribution to id will make use of 
   * contactConductance. This function will be called
   * with the dcDeltaSolution vector instead of the incremental quantities
   */
  ipPrime = contactConductance( pDevice, pPContact, deltaVContact,
			       solution, tranAnalysis, intCoeff );
  
  ipPrime *= JNorm * pDevice->width * LNorm;
  ip += ipPrime;
  *id = ip;
}

void 
   NBJT2current(TWOdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff, 
                double *ie, double *ic)
{
  TWOcontact *pEmitContact = pDevice->pLastContact;
  TWOcontact *pColContact = pDevice->pFirstContact;
/*  TWOcontact *pBaseContact = pDevice->pFirstContact->next; */
  double *solution, iePrime, icPrime;
/*  double ib; */
  
  solution = pDevice->dcDeltaSolution;
  
  *ie = contactCurrent( pDevice, pEmitContact );
  *ic = contactCurrent( pDevice, pColContact );
  /*
  ib = contactCurrent( pDevice, pBaseContact );
  fprintf(stdout, "BJT current: ( %11.4e error )\n", *ic+ib+*ie );
  fprintf(stdout, "     Ic = %11.4e     Ib = %11.4e\n", *ic, ib );
  fprintf(stdout, "     Ie = %11.4e\n", *ie );
  */
  
  iePrime = contactConductance( pDevice, pEmitContact, FALSE, solution,
			       tranAnalysis, intCoeff );
  
  icPrime = contactConductance( pDevice, pColContact, FALSE, solution,
			       tranAnalysis, intCoeff );
  
  iePrime *= JNorm * pDevice->width * LNorm;
  icPrime *= JNorm * pDevice->width * LNorm;
  
  *ie += iePrime;
  *ic += icPrime;
}

void 
   NUMOScurrent(TWOdevice *pDevice, BOOLEAN tranAnalysis, double *intCoeff, 
                double *id, double *is, double *ig)
{
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
/*  TWOcontact *pBContact = pDevice->pLastContact; */
  double *solution, idPrime, isPrime, igPrime;
/*  double ib; */
  
  solution = pDevice->dcDeltaSolution;
  
  *id = contactCurrent( pDevice, pDContact );

/* 
 * This is a terrible hack
 */  
  
#ifdef NORMAL_GATE  
  *ig = GateTypeCurrent( pDevice, pGContact, tranAnalysis );
#else
  *ig = GateTypeCurrent( pDevice, pGContact);
#endif  
  
  *is = contactCurrent( pDevice, pSContact );
  /*
  ib = contactCurrent( pDevice, pBContact );
  fprintf(stdout, "MOS current: ( %11.4e error )\n", *id+*ig+*is+ib );
  fprintf(stdout, "     Id = %11.4e     Is = %11.4e\n", *id, *is );
  fprintf(stdout, "     Ig = %11.4e     Ib = %11.4e\n", *ig, ib );
  */

  idPrime = contactConductance( pDevice, pDContact, FALSE,
			       solution, tranAnalysis, intCoeff );
  
  isPrime = contactConductance( pDevice, pSContact, FALSE,
			       solution, tranAnalysis, intCoeff );
  
  igPrime = GateTypeConductance( pDevice, pGContact, FALSE,
				solution, tranAnalysis, intCoeff );
  
  idPrime *= JNorm * pDevice->width * LNorm;
  isPrime *= JNorm * pDevice->width * LNorm;
  igPrime *= JNorm * pDevice->width * LNorm;
  
  *id += idPrime;
  *is += isPrime;
  *ig += igPrime;
}
