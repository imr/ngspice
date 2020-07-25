/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/**********
Copyright 2020 Anamosic Ballenegger Design
Author: 2020 Florian Ballenegger
Vector version.
**********/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/SIMD/simdvector.h"
#include "ngspice/SIMD/simdop.h"
#include "ngspice/SIMD/simdckt.h"

void
vecN_CKTterr(VecNi qcap, CKTcircuit *ckt, double *timeStep)
{ 
    VecNd volttol;
    VecNd chargetol;
    VecNd tol;
    VecNd del;
    VecNd diff[8];
    double deltmp[8];
    double factor=0;
    VecNi ccap;
    int i;
    int j;
    int k;
    static double gearCoeff[] = {
        .5,
        .2222222222,
        .1363636364,
        .096,
        .07299270073,
        .05830903790
    };
    static double trapCoeff[] = {
        .5,
        .08333333333
    };
    ccap = qcap+1;

    volttol = ckt->CKTabstol + ckt->CKTreltol * 
            vecN_MAX( vecN_fabs(vecN_lu(ckt->CKTstate0,ccap)), vecN_fabs(vecN_lu(ckt->CKTstate1,ccap)));
            
    chargetol = vecN_MAX(vecN_fabs(vecN_lu(ckt->CKTstate0,qcap)),vecN_fabs(vecN_lu(ckt->CKTstate1,qcap)));
    chargetol = ckt->CKTreltol * vecN_MAX(chargetol,vecN_broadcast(ckt->CKTchgtol))/ckt->CKTdelta;
    tol = vecN_MAX(volttol,chargetol);
    
    /* now divided differences */
    for(i=ckt->CKTorder+1;i>=0;i--) {
        diff[i] = vecN_lu(ckt->CKTstates[i],qcap);
    }
    for(i=0 ; i <= ckt->CKTorder ; i++) {
        deltmp[i] = ckt->CKTdeltaOld[i];
    }
    j = ckt->CKTorder;
    for (;;) {
        for(i=0;i <= j;i++) {
            diff[i] = (diff[i] - diff[i+1])/deltmp[i];
        }
        if (--j < 0) break;
        for(i=0;i <= j;i++) {
            deltmp[i] = deltmp[i+1] + ckt->CKTdeltaOld[i];
        }
    }
    switch(ckt->CKTintegrateMethod) {
        case GEAR:
            factor = gearCoeff[ckt->CKTorder-1];
            break;

        case TRAPEZOIDAL:
            factor = trapCoeff[ckt->CKTorder - 1] ;
            break;
    }
    del = ckt->CKTtrtol * tol/vecN_MAX(vecN_broadcast(ckt->CKTabstol),factor * vecN_fabs(diff[0]));
    if(ckt->CKTorder == 2) {
        del = vecN_sqrt(del);
    } else if (ckt->CKTorder > 2) {
        del = vecN_exp(vecN_log(del)/ckt->CKTorder);
    }
    /* now reduce with minimum */
    for(k=0;k<NSIMD;k++)
    if(del[k]<(*timeStep))
    	*timeStep = del[k];
    return;
}
