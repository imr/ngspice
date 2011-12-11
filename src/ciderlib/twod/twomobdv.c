/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/twomesh.h"
#include "ngspice/bool.h"
#include "twoddefs.h"
#include "twodext.h"

/*
 * Load the derivatives of the current with respect to changes in
 * the mobility, for all edges of a silicon element.
 * It is known a priori that the element is made of semiconductor.
 * These routines work for both channel and bulk elements.
 */

void
  TWO_mobDeriv( TWOelem* pElem, int chanType, double ds )
	/* TWOelem *pElem: channel or bulk element              */
	/* int chanType: flag for direction of channel          */
	/* double ds: normalized hgt (len) of interface element */
{
  TWOnode *pNode;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  BOOLEAN channel = pElem->channel;
  double dx, dy, rDx, rDy;
  double coeffHx, coeffHy, coeffHs = 0.0;
  double coeffVx, coeffVy, coeffVs = 0.0;
  double dFnxDMun, dFnyDMun, dFnDEs = 0.0;
  double dFpxDMup, dFpyDMup, dFpDEs = 0.0;
  double dMnDEs, dMnDEx, dMnDEy, dMnDWx, dMnDWy;
  double dMpDEs, dMpDEx, dMpDEy, dMpDWx, dMpDWy;
  
  /* Initialize various quantities */
  dx = pElem->dx;
  dy = pElem->dy;
  rDx = 1.0 / dx;
  rDy = 1.0 / dy;
  
  /* compute length-dependent parameters */
  coeffHx = 0.25 * dy * rDx;	/* For horizontal edges */
  coeffHy = 0.25;
  coeffVx = 0.25;		/* For vertical edges */
  coeffVy = 0.25 * dx * rDy;
  switch ( chanType ) {
  case 0:
  case 3:
    coeffHs = 0.25 * dy / ds;
    coeffVs = 0.25 * dx / ds;
    break;
  case 1:
  case 2:
    coeffHs = - 0.25 * dy / ds;
    coeffVs = - 0.25 * dx / ds;
    break;
  }
  
  /* Get pointers to element's edges */
  pTEdge = pElem->pTopEdge;
  pBEdge = pElem->pBotEdge;
  pLEdge = pElem->pLeftEdge;
  pREdge = pElem->pRightEdge;
  
  /* Get element mobility derivatives for fast access later */
  dMnDEs = pElem->dMunDEs;
  dMnDEx = pElem->dMunDEx;
  dMnDEy = pElem->dMunDEy;
  dMnDWx = pElem->dMunDWx;
  dMnDWy = pElem->dMunDWy;
  dMpDEs = pElem->dMupDEs;
  dMpDEx = pElem->dMupDEx;
  dMpDEy = pElem->dMupDEy;
  dMpDWx = pElem->dMupDWx;
  dMpDWy = pElem->dMupDWy;
  
  /* Add mobility derivatives due to Top Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffHx * (pTEdge->wdfn * rDx);
  dFnyDMun = coeffHy * (pTEdge->wdfn * rDx);
  dFpxDMup = coeffHx * (pTEdge->wdfp * rDx);
  dFpyDMup = coeffHy * (pTEdge->wdfp * rDx);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffHs * (pTEdge->wdfn * rDx) * dMnDEs;
    dFpDEs   = coeffHs * (pTEdge->wdfp * rDx) * dMpDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1jP1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1jP1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNjP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1jP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPjP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1jP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  
  /* Add mobility derivatives due to Bottom Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffHx * (pBEdge->wdfn * rDx);
  dFnyDMun = coeffHy * (pBEdge->wdfn * rDx);
  dFpxDMup = coeffHx * (pBEdge->wdfp * rDx);
  dFpyDMup = coeffHy * (pBEdge->wdfp * rDx);
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsijM1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jM1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsijM1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jM1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNjM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1jM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPjM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1jM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if (channel ) {
    dFnDEs   = coeffHs * (pBEdge->wdfn * rDx) * dMnDEs;
    dFpDEs   = coeffHs * (pBEdge->wdfp * rDx) * dMpDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1jM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsijM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1jM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsijM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  /* Add mobility derivatives due to Left Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffVx * (pLEdge->wdfn * rDy);
  dFnyDMun = coeffVy * (pLEdge->wdfn * rDy);
  dFpxDMup = coeffVx * (pLEdge->wdfp * rDy);
  dFpyDMup = coeffVy * (pLEdge->wdfp * rDy);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffVs * (pLEdge->wdfn * rDy) * dMnDEs;
    dFpDEs   = coeffVs * (pLEdge->wdfp * rDy) * dMpDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsijM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsijM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiInP1) += dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
    *(pNode->fNPsiOxP1) -= dFnDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiInP1) += dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
    *(pNode->fPPsiOxP1) -= dFpDEs;
  }
  
  /* Add mobility derivatives due to Right Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffVx * (pREdge->wdfn * rDy);
  dFnyDMun = coeffVy * (pREdge->wdfn * rDy);
  dFpxDMup = coeffVx * (pREdge->wdfp * rDy);
  dFpyDMup = coeffVy * (pREdge->wdfp * rDy);
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsi) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1jP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsi) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1jP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffVs * (pREdge->wdfn * rDy) * dMnDEs;
    dFpDEs   = coeffVs * (pREdge->wdfp * rDy) * dMpDEs;
    *(pNode->fNPsiInM1) -= dFnDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiOxM1) += dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fPPsiInM1) -= dFpDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiOxM1) += dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1jM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsijM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1jM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsijM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  return;
}

void
  TWONmobDeriv(TWOelem *pElem, int chanType, double ds)
	/* TWOelem *pElem: channel or bulk element              */
	/* int chanType: flag for direction of channel          */
	/* double ds: normalized hgt (len) of interface element */
{
  TWOnode *pNode;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  BOOLEAN channel = pElem->channel;
  double dx, dy, rDx, rDy;
  double coeffHx, coeffHy, coeffHs = 0.0;
  double coeffVx, coeffVy, coeffVs = 0.0;
  double dFnxDMun, dFnyDMun, dFnDEs = 0.0;
  double dMnDEs, dMnDEx, dMnDEy, dMnDWx, dMnDWy;
  
  /* Initialize various quantities */
  dx = pElem->dx;
  dy = pElem->dy;
  rDx = 1.0 / dx;
  rDy = 1.0 / dy;
  
  /* compute length-dependent parameters */
  coeffHx = 0.25 * dy * rDx;	/* For horizontal edges */
  coeffHy = 0.25;
  coeffVx = 0.25;		/* For vertical edges */
  coeffVy = 0.25 * dx * rDy;
  switch ( chanType ) {
  case 0:
  case 3:
    coeffHs = 0.25 * dy / ds;
    coeffVs = 0.25 * dx / ds;
    break;
  case 1:
  case 2:
    coeffHs = - 0.25 * dy / ds;
    coeffVs = - 0.25 * dx / ds;
    break;
  }
  
  /* Get pointers to element's edges */
  pTEdge = pElem->pTopEdge;
  pBEdge = pElem->pBotEdge;
  pLEdge = pElem->pLeftEdge;
  pREdge = pElem->pRightEdge;
  
  /* Get element mobility derivatives for fast access later */
  dMnDEs = pElem->dMunDEs;
  dMnDEx = pElem->dMunDEx;
  dMnDEy = pElem->dMunDEy;
  dMnDWx = pElem->dMunDWx;
  dMnDWy = pElem->dMunDWy;
  
  /* Add mobility derivatives due to Top Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffHx * (pTEdge->wdfn * rDx);
  dFnyDMun = coeffHy * (pTEdge->wdfn * rDx);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffHs * (pTEdge->wdfn * rDx) * dMnDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
  }
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1jP1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );

  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNjP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1jP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
  }
  
  
  /* Add mobility derivatives due to Bottom Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffHx * (pBEdge->wdfn * rDx);
  dFnyDMun = coeffHy * (pBEdge->wdfn * rDx);
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsijM1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jM1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNjM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1jM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  
  /* both continuity wrto surface potential derivatives */
  if (channel ) {
    dFnDEs   = coeffHs * (pBEdge->wdfn * rDx) * dMnDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1jM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsijM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
  }
  
  /* Add mobility derivatives due to Left Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffVx * (pLEdge->wdfn * rDy);
  dFnyDMun = coeffVy * (pLEdge->wdfn * rDy);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsi) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );

  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffVs * (pLEdge->wdfn * rDy) * dMnDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiInP1) -= dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
    *(pNode->fNPsiOxP1) += dFnDEs;
  }
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsijM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1jM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiP1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNiP1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNiP1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );

  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiInP1) += dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
    *(pNode->fNPsiOxP1) -= dFnDEs;
  }
  
  /* Add mobility derivatives due to Right Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFnxDMun = coeffVx * (pREdge->wdfn * rDy);
  dFnyDMun = coeffVy * (pREdge->wdfn * rDy);
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsi) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsijP1) +=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1jP1) +=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNN) +=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNNjP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1jP1) +=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );

  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFnDEs   = coeffVs * (pREdge->wdfn * rDy) * dMnDEs;
    *(pNode->fNPsiInM1) -= dFnDEs;
    *(pNode->fNPsiIn)   -= dFnDEs;
    *(pNode->fNPsiOxM1) += dFnDEs;
    *(pNode->fNPsiOx)   += dFnDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* n continuity wrto potential derivatives */
  *(pNode->fNPsiiM1jM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  *(pNode->fNPsijM1) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pTEdge->dWnDpsiP1 )
      + dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsi) -=
    - dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pREdge->dWnDpsiP1 );
  *(pNode->fNPsiiM1) -=
    dFnxDMun * ( dMnDEx - dMnDWx * pBEdge->dWnDpsiP1 )
      - dFnyDMun * ( dMnDEy - dMnDWy * pLEdge->dWnDpsiP1 );
  /* n continuity wrto n derivatives */
  *(pNode->fNNiM1jM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDn );
  *(pNode->fNNjM1) -=
    dFnxDMun * ( dMnDWx * pTEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDn );
  *(pNode->fNN) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDnP1 )
      + dFnyDMun * ( dMnDWy * pREdge->dWnDnP1 );
  *(pNode->fNNiM1) -=
    dFnxDMun * ( dMnDWx * pBEdge->dWnDn )
      + dFnyDMun * ( dMnDWy * pLEdge->dWnDnP1 );

  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fNPsiInM1) += dFnDEs;
    *(pNode->fNPsiIn)   += dFnDEs;
    *(pNode->fNPsiOxM1) -= dFnDEs;
    *(pNode->fNPsiOx)   -= dFnDEs;
  }
  
  return;
}

void
  TWOPmobDeriv( TWOelem *pElem, int chanType, double ds )
	/* TWOelem *pElem: channel or bulk element             */
	/*int chanType: flag for direction of channel          */
	/*double ds: normalized hgt (len) of interface element */
{
  TWOnode *pNode;
  TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
  BOOLEAN channel = pElem->channel;
  double dx, dy, rDx, rDy;
  double coeffHx, coeffHy, coeffHs = 0.0;
  double coeffVx, coeffVy, coeffVs = 0.0;
  double dFpxDMup, dFpyDMup, dFpDEs = 0.0;
  double dMpDEs, dMpDEx, dMpDEy, dMpDWx, dMpDWy;
  
  /* Initialize various quantities */
  dx = pElem->dx;
  dy = pElem->dy;
  rDx = 1.0 / dx;
  rDy = 1.0 / dy;
  
  /* compute length-dependent parameters */
  coeffHx = 0.25 * dy * rDx;	/* For horizontal edges */
  coeffHy = 0.25;
  coeffVx = 0.25;		/* For vertical edges */
  coeffVy = 0.25 * dx * rDy;
  switch ( chanType ) {
  case 0:
  case 3:
    coeffHs = 0.25 * dy / ds;
    coeffVs = 0.25 * dx / ds;
    break;
  case 1:
  case 2:
    coeffHs = - 0.25 * dy / ds;
    coeffVs = - 0.25 * dx / ds;
    break;
  }
  
  /* Get pointers to element's edges */
  pTEdge = pElem->pTopEdge;
  pBEdge = pElem->pBotEdge;
  pLEdge = pElem->pLeftEdge;
  pREdge = pElem->pRightEdge;
  
  /* Get element mobility derivatives for fast access later */
  dMpDEs = pElem->dMupDEs;
  dMpDEx = pElem->dMupDEx;
  dMpDEy = pElem->dMupDEy;
  dMpDWx = pElem->dMupDWx;
  dMpDWy = pElem->dMupDWy;
  
  /* Add mobility derivatives due to Top Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFpxDMup = coeffHx * (pTEdge->wdfp * rDx);
  dFpyDMup = coeffHy * (pTEdge->wdfp * rDx);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFpDEs   = coeffHs * (pTEdge->wdfp * rDx) * dMpDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1jP1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPjP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1jP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  
  /* Add mobility derivatives due to Bottom Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFpxDMup = coeffHx * (pBEdge->wdfp * rDx);
  dFpyDMup = coeffHy * (pBEdge->wdfp * rDx);
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsijM1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jM1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPjM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1jM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if (channel ) {
    dFpDEs   = coeffHs * (pBEdge->wdfp * rDx) * dMpDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1jM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsijM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  /* Add mobility derivatives due to Left Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFpxDMup = coeffVx * (pLEdge->wdfp * rDy);
  dFpyDMup = coeffVy * (pLEdge->wdfp * rDy);
  
  /* Do Top-Left (TL) Node of Element */
  pNode = pElem->pTLNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsi) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFpDEs   = coeffVs * (pLEdge->wdfp * rDy) * dMpDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiInP1) -= dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
    *(pNode->fPPsiOxP1) += dFpDEs;
  }
  
  /* Do Bottom-Left (BL) Node of Element */
  pNode = pElem->pBLNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsijM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1jM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiP1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPiP1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPiP1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiInP1) += dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
    *(pNode->fPPsiOxP1) -= dFpDEs;
  }
  
  /* Add mobility derivatives due to Right Edge */
  /* First compute derivatives of cont. eqn's */ 
  dFpxDMup = coeffVx * (pREdge->wdfp * rDy);
  dFpyDMup = coeffVy * (pREdge->wdfp * rDy);
  
  /* Do Top-Right (TR) Node of Element */
  pNode = pElem->pTRNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsi) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsijP1) +=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1jP1) +=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPP) +=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPPjP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1jP1) +=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );
  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    dFpDEs   = coeffVs * (pREdge->wdfp * rDy) * dMpDEs;
    *(pNode->fPPsiInM1) -= dFpDEs;
    *(pNode->fPPsiIn)   -= dFpDEs;
    *(pNode->fPPsiOxM1) += dFpDEs;
    *(pNode->fPPsiOx)   += dFpDEs;
  }
  
  /* Do Bottom-Right (BR) Node of Element */
  pNode = pElem->pBRNode;
  /* p continuity wrto potential derivatives */
  *(pNode->fPPsiiM1jM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  *(pNode->fPPsijM1) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pTEdge->dWpDpsiP1 )
      + dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsi) -=
    - dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pREdge->dWpDpsiP1 );
  *(pNode->fPPsiiM1) -=
    dFpxDMup * ( dMpDEx - dMpDWx * pBEdge->dWpDpsiP1 )
      - dFpyDMup * ( dMpDEy - dMpDWy * pLEdge->dWpDpsiP1 );
  /* p continuity wrto p derivatives */
  *(pNode->fPPiM1jM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDp );
  *(pNode->fPPjM1) -=
    dFpxDMup * ( dMpDWx * pTEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDp );
  *(pNode->fPP) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDpP1 )
      + dFpyDMup * ( dMpDWy * pREdge->dWpDpP1 );
  *(pNode->fPPiM1) -=
    dFpxDMup * ( dMpDWx * pBEdge->dWpDp )
      + dFpyDMup * ( dMpDWy * pLEdge->dWpDpP1 );

  /* both continuity wrto surface potential derivatives */
  if ( channel ) {
    *(pNode->fPPsiInM1) += dFpDEs;
    *(pNode->fPPsiIn)   += dFpDEs;
    *(pNode->fPPsiOxM1) -= dFpDEs;
    *(pNode->fPPsiOx)   -= dFpDEs;
  }
  
  return;
}
