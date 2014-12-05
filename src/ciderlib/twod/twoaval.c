/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1990 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "twoddefs.h"
#include "twodext.h"


double
TWOavalanche(TWOelem *pElem, TWOnode *pNode)
{

    TWOelem *pElemTL, *pElemTR, *pElemBL, *pElemBR;
    TWOedge *pEdgeT, *pEdgeB, *pEdgeL, *pEdgeR;
    int materT = 0 , materB = 0, materL = 0 , materR = 0;
    double dxL = 0.0, dxR = 0.0, dyT = 0.0, dyB = 0.0;
    double ef1, ef2, coeff1, coeff2;
    double enx, eny, epx, epy, jnx, jny, jpx, jpy;
    double current, eField; 
    double generation = 0.0;
    double eiip2 = 4.0e5 / ENorm;
    double aiip2 = 6.71e5 * LNorm;
    double biip2 = 1.693e6 / ENorm;
    TWOmaterial *info = pElem->matlInfo;

    /* Find all four neighboring elements */
    pElemTL = pNode->pTLElem;
    pElemTR = pNode->pTRElem;
    pElemBL = pNode->pBLElem;
    pElemBR = pNode->pBRElem;

    /* Null edge pointers */
    pEdgeT = pEdgeB = pEdgeL = pEdgeR = NULL;

    /* Find edges next to node */
    if ( pElemTL != NULL ) {
      if (pElemTL->evalEdges[1]) {
	pEdgeT = pElemTL->pRightEdge;
	materT = pElemTL->elemType;
	dyT = pElemTL->dy;
      }
      if (pElemTL->evalEdges[2]) {
	pEdgeL = pElemTL->pBotEdge;
	materL = pElemTL->elemType;
	dxL = pElemTL->dx;
      }
    }
    if ( pElemTR != NULL ) {
      if (pElemTR->evalEdges[3]) {
	pEdgeT = pElemTR->pLeftEdge;
	materT = pElemTR->elemType;
	dyT = pElemTR->dy;
      }
      if (pElemTR->evalEdges[2]) {
	pEdgeR = pElemTR->pBotEdge;
	materR = pElemTR->elemType;
	dxR = pElemTR->dx;
      }
    }
    if ( pElemBR != NULL ) {
      if (pElemBR->evalEdges[3]) {
	pEdgeB = pElemBR->pLeftEdge;
	materB = pElemBR->elemType;
	dyB = pElemBR->dy;
      }
      if (pElemBR->evalEdges[0]) {
	pEdgeR = pElemBR->pTopEdge;
	materR = pElemBR->elemType;
	dxR = pElemBR->dx;
      }
    }
    if ( pElemBL != NULL ) {
      if (pElemBL->evalEdges[1]) {
	pEdgeB = pElemBL->pRightEdge;
	materB = pElemBL->elemType;
	dyB = pElemBL->dy;
      }
      if (pElemBL->evalEdges[0]) {
	pEdgeL = pElemBL->pTopEdge;
	materL = pElemBL->elemType;
	dxL = pElemBL->dx;
      }
    }

    /* compute horizontal vector components */
    /* No more than one of Left Edge or Right Edge is absent */
    /* If one is absent the other is guaranteed to be from silicon */
    if (pEdgeL == NULL) {
      if ( pNode->nodeType == CONTACT ) {
	enx = -(pEdgeR->dPsi + pEdgeR->dCBand) / dxR;
	epx = -(pEdgeR->dPsi - pEdgeR->dVBand) / dxR;
	jnx = pEdgeR->jn;
	jpx = pEdgeR->jp;
      } else {
	enx = 0.0;
	epx = 0.0;
	jnx = 0.0;
	jpx = 0.0;
      }
    } else if (pEdgeR == NULL) {
      if ( pNode->nodeType == CONTACT ) {
	enx = -(pEdgeL->dPsi + pEdgeL->dCBand) / dxL;
	epx = -(pEdgeL->dPsi - pEdgeL->dVBand) / dxL;
	jnx = pEdgeL->jn;
	jpx = pEdgeL->jp;
      } else {
	enx = 0.0;
	epx = 0.0;
	jnx = 0.0;
	jpx = 0.0;
      }
    } else { /* Both edges are present */
      coeff1 = dxL / (dxL + dxR);
      coeff2 = dxR / (dxL + dxR);
      ef1 = -(pEdgeL->dPsi + pEdgeL->dCBand) / dxL;
      ef2 = -(pEdgeR->dPsi + pEdgeR->dCBand) / dxR;
      enx = coeff2 * ef1 + coeff1 * ef2;
      ef1 = -(pEdgeL->dPsi - pEdgeL->dVBand) / dxL;
      ef2 = -(pEdgeR->dPsi - pEdgeR->dVBand) / dxR;
      epx = coeff2 * ef1 + coeff1 * ef2;
      if ( (materL == INSULATOR) || (materR == INSULATOR) ) {
	jnx = 0.0;
	jpx = 0.0;
      } else {
        jnx = coeff2 * pEdgeL->jn + coeff1 * pEdgeR->jn;
        jpx = coeff2 * pEdgeL->jp + coeff1 * pEdgeR->jp;
      }
    }

    /* compute vertical vector components */
    /* No more than one of Top Edge or Bottom Edge is absent */
    /* If one is absent the other is guaranteed to be from silicon */
    if (pEdgeT == NULL) {
      if ( pNode->nodeType == CONTACT ) {
	eny = -(pEdgeB->dPsi + pEdgeB->dCBand) / dyB;
	epy = -(pEdgeB->dPsi - pEdgeB->dVBand) / dyB;
	jny = pEdgeB->jn;
	jpy = pEdgeB->jp;
      } else {
	eny = 0.0;
	epy = 0.0;
	jny = 0.0;
	jpy = 0.0;
      }
    } else if (pEdgeB == NULL) {
      if ( pNode->nodeType == CONTACT ) {
	eny = -(pEdgeT->dPsi + pEdgeT->dCBand) / dyT;
	epy = -(pEdgeT->dPsi - pEdgeT->dVBand) / dyT;
	jny = pEdgeT->jn;
	jpy = pEdgeT->jp;
      } else {
	eny = 0.0;
	epy = 0.0;
	jny = 0.0;
	jpy = 0.0;
      }
    } else { /* Both edges are present */
      coeff1 = dyT / (dyT + dyB);
      coeff2 = dyB / (dyT + dyB);
      ef1 = -(pEdgeT->dPsi + pEdgeT->dCBand) / dyT;
      ef2 = -(pEdgeB->dPsi + pEdgeB->dCBand) / dyB;
      eny = coeff2 * ef1 + coeff1 * ef2;
      ef1 = -(pEdgeT->dPsi - pEdgeT->dVBand) / dyT;
      ef2 = -(pEdgeB->dPsi - pEdgeB->dVBand) / dyB;
      epy = coeff2 * ef1 + coeff1 * ef2;
      if ( (materT == INSULATOR) || (materB == INSULATOR) ) {
	jny = 0.0;
	jpy = 0.0;
      } else {
        jny = coeff2 * pEdgeT->jn + coeff1 * pEdgeB->jn;
        jpy = coeff2 * pEdgeT->jp + coeff1 * pEdgeB->jp;
      }
    }

/*
    fprintf(stderr,"%% en = (%9.2e,%9.2e), jn = (%9.2e,%9.2e)\n",
      enx,eny,jnx,jny);
    fprintf(stderr,"%% ep = (%9.2e,%9.2e), jp = (%9.2e,%9.2e)\n",
      epx,epy,jpx,jpy);
*/

    /* now calculate the avalanche generation rate */
    current = hypot(jnx, jny);
    if ( current != 0.0 ) {
	eField = (enx * jnx + eny * jny) / current;
	if ( (eField > 0) && ( info->bii[ELEC] / eField <= 80.0) ) {
	    generation += current * info->aii[ELEC]
		    * exp( - info->bii[ELEC] / eField );
	}
    }
    current = hypot(jpx, jpy);
    if ( current != 0.0 ) {
	eField = (epx * jpx + epy * jpy) / current;
	if ( eField > eiip2 ) {
	    generation += current * aiip2 * exp( - biip2 / eField );
	} else if ( (eField > 0) && ( info->bii[HOLE] / eField <= 80.0) ) {
	    generation += current * info->aii[HOLE]
		    * exp( - info->bii[HOLE] / eField );
	}
    }
    return( generation );
}
