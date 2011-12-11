/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1990 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/twomesh.h"
#include "twoddefs.h"
#include "twodext.h"

/*
 * Compute the 2-D field-dependent mobility at the center of an element.
 * It is known a priori that the element belongs to a semiconductor domain.
 */
				
void
TWO_mobility(TWOelem *pElem, double eSurf)
{

    TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
    double dx, dy, rDx, rDy;
    double enx, eny, wnx, wny, concav;
    double epx, epy, wpx, wpy;

    /* Initialize various quantities */
    dx = pElem->dx;
    dy = pElem->dy;
    rDx = 0.5 / dx;		/* Includes averaging factor of 0.5 */
    rDy = 0.5 / dy;		/* Includes averaging factor of 0.5 */

    /* Get pointers to element's edges */
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;

    /* Calculate electric field at element center */
    enx = -rDx *(pTEdge->dPsi + pTEdge->dCBand + pBEdge->dPsi + pBEdge->dCBand);
    epx = -rDx *(pTEdge->dPsi - pTEdge->dVBand + pBEdge->dPsi - pBEdge->dVBand);
    eny = -rDy *(pLEdge->dPsi + pLEdge->dCBand + pREdge->dPsi + pREdge->dCBand);
    epy = -rDy *(pLEdge->dPsi - pLEdge->dVBand + pREdge->dPsi - pREdge->dVBand);

    /* Calculate weighted carrier driving force at element center */
    wnx = rDx * (pTEdge->wdfn + pBEdge->wdfn);
    wpx = rDx * (pTEdge->wdfp + pBEdge->wdfp);
    wny = rDy * (pLEdge->wdfn + pREdge->wdfn);
    wpy = rDy * (pLEdge->wdfp + pREdge->wdfp);

    /* compute the mobility for the element */
    /* Average concentrations at the four corners */
    concav = 0.25 * ( pElem->pTLNode->totalConc + pElem->pTRNode->totalConc +
        pElem->pBLNode->totalConc + pElem->pBRNode->totalConc );
    MOBsurfElec(pElem->matlInfo, pElem, enx, eny, eSurf, wnx, wny, concav);
    MOBsurfHole(pElem->matlInfo, pElem, epx, epy, eSurf, wpx, wpy, concav);
    return;
}

void
TWONmobility(TWOelem *pElem, double eSurf)
{

    TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
    double dx, dy, rDx, rDy;
    double enx, eny, wnx, wny, concav;

    /* Initialize various quantities */
    dx = pElem->dx;
    dy = pElem->dy;
    rDx = 0.5 / dx;		/* Includes averaging factor of 0.5 */
    rDy = 0.5 / dy;		/* Includes averaging factor of 0.5 */

    /* Get pointers to element's edges */
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;

    /* Calculate electric field at element center */
    enx = -rDx *(pTEdge->dPsi + pTEdge->dCBand + pBEdge->dPsi + pBEdge->dCBand);
    eny = -rDy *(pLEdge->dPsi + pLEdge->dCBand + pREdge->dPsi + pREdge->dCBand);

    /* Calculate weighted carrier driving force at element center */
    wnx = rDx * (pTEdge->wdfn + pBEdge->wdfn);
    wny = rDy * (pLEdge->wdfn + pREdge->wdfn);

    /* compute the mobility for the element */
    /* Average concentrations at the four corners */
    concav = 0.25 * ( pElem->pTLNode->totalConc + pElem->pTRNode->totalConc +
        pElem->pBLNode->totalConc + pElem->pBRNode->totalConc );
    MOBsurfElec(pElem->matlInfo, pElem, enx, eny, eSurf, wnx, wny, concav);

    return;
}

void
TWOPmobility(TWOelem *pElem, double eSurf)
{

    TWOedge *pTEdge, *pBEdge, *pLEdge, *pREdge;
    double dx, dy, rDx, rDy;
    double epx, epy, wpx, wpy, concav;

    /* Initialize various quantities */
    dx = pElem->dx;
    dy = pElem->dy;
    rDx = 0.5 / dx;		/* Includes averaging factor of 0.5 */
    rDy = 0.5 / dy;		/* Includes averaging factor of 0.5 */

    /* Get pointers to element's edges */
    pTEdge = pElem->pTopEdge;
    pBEdge = pElem->pBotEdge;
    pLEdge = pElem->pLeftEdge;
    pREdge = pElem->pRightEdge;

    /* Calculate electric field at element center */
    epx = -rDx *(pTEdge->dPsi - pTEdge->dVBand + pBEdge->dPsi - pBEdge->dVBand);
    epy = -rDy *(pLEdge->dPsi - pLEdge->dVBand + pREdge->dPsi - pREdge->dVBand);

    /* Calculate weighted carrier driving force at element center */
    wpx = rDx * (pTEdge->wdfp + pBEdge->wdfp);
    wpy = rDy * (pLEdge->wdfp + pREdge->wdfp);

    /* compute the mobility for the element */
    /* Average concentrations at the four corners */
    concav = 0.25 * ( pElem->pTLNode->totalConc + pElem->pTRNode->totalConc +
        pElem->pBLNode->totalConc + pElem->pBRNode->totalConc );
    MOBsurfHole(pElem->matlInfo, pElem, epx, epy, eSurf, wpx, wpy, concav);

    return;
}
