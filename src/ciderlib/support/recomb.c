/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/cidersupt.h"

/*
 * function recomb calculates the recobination rates and the 
 * derivatives with respect to n and p. SRH recombination is 
 * always assumed. Auger recombination is specified by the flag
 * 'Auger' which by default is false.
 */

void
recomb (double ni, double pi, double tn, double tp, double cn, double cp, 
        double nie, double *pUnet, double *pDuDn, double *pDuDp)
{
    double uSrh, uSrhNum, uSrhDen, perUdenSq, duSrhDn, duSrhDp;
    double cncp, uAug, duAugDn, duAugDp, uNet, duDn, duDp;

    uSrhNum = ni * pi - nie * nie;
    uSrhDen = tp * (ni + nie) + tn * (pi + nie);
    uSrh = uSrhNum / uSrhDen;
    perUdenSq = 1.0 / (uSrhDen * uSrhDen);
    duSrhDn = (pi * uSrhDen - uSrhNum * tp) * perUdenSq;
    duSrhDp = (ni * uSrhDen - uSrhNum * tn) * perUdenSq;
    
    
    if (Auger && uSrhNum >= 0.0) { /* XXX Auger Global */
	cncp = cn * ni + cp * pi;
	uAug = cncp * uSrhNum;
	duAugDn = cn * uSrhNum + cncp * pi;
	duAugDp = cp * uSrhNum + cncp * ni;
	uNet = uSrh + uAug;
	duDn = duSrhDn + duAugDn;
	duDp = duSrhDp + duAugDp;
    }
    else {
	uNet = uSrh;
	duDn = duSrhDn;
	duDp = duSrhDp;
    }

/* return uNet duDn duDp */
    *pUnet = uNet;
    *pDuDn = duDn;
    *pDuDp = duDp;
}
