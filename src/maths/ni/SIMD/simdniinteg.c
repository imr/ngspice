/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/**********
SIMD version:
Copyright 2020 Anamosic Ballenegger Design.
Author: 2020 Florian Ballenegger
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"

#include "ngspice/SIMD/simdniinteg.h"
#include "ngspice/SIMD/simdop.h"

int
vecN_NIintegrate(CKTcircuit *ckt, double *geq, double *ceq, double cap, VecNm qcap)
{
    static char *ordmsg = "Illegal integration order";
    static char *methodmsg = "Unknown integration method";
    
    VecNm ccap = qcap + 1;
    
    switch(ckt->CKTintegrateMethod) {

    case TRAPEZOIDAL:
        switch(ckt->CKTorder) {
        case 1:
	    vecN_StateStore(ckt->CKTstate0, ccap, ckt->CKTag[0] * vecN_StateAccess(ckt->CKTstate0,qcap) 
                    + ckt->CKTag[1] * vecN_StateAccess(ckt->CKTstate1,qcap));
            break;
        case 2:
	    vecN_StateStore(ckt->CKTstate0, ccap, - vecN_StateAccess(ckt->CKTstate1,ccap) * ckt->CKTag[1] + 
                    ckt->CKTag[0] * 
                    ( vecN_StateAccess(ckt->CKTstate0,qcap) - vecN_StateAccess(ckt->CKTstate1,qcap) ) );
            break;
        default:
            errMsg = TMALLOC(char, strlen(ordmsg) + 1);
            strcpy(errMsg,ordmsg);
            return(E_ORDER);
        }
        break;
    case GEAR:
        {
	VecNd nstateccap = vecN_SIMDTOVECTOR(0);
        switch(ckt->CKTorder) {

        case 6:
            nstateccap += ckt->CKTag[6]* vecN_StateAccess(ckt->CKTstate6,qcap);
            /* fall through */
        case 5:
            nstateccap += ckt->CKTag[5]* vecN_StateAccess(ckt->CKTstate5,qcap);
            /* fall through */
        case 4:
            nstateccap += ckt->CKTag[4]* vecN_StateAccess(ckt->CKTstate4,qcap);
            /* fall through */
        case 3:
            nstateccap += ckt->CKTag[3]* vecN_StateAccess(ckt->CKTstate3,qcap);
            /* fall through */
        case 2:
            nstateccap += ckt->CKTag[2]* vecN_StateAccess(ckt->CKTstate2,qcap);
            /* fall through */
        case 1:
            nstateccap += ckt->CKTag[1]* vecN_StateAccess(ckt->CKTstate1,qcap);
            nstateccap += ckt->CKTag[0]* vecN_StateAccess(ckt->CKTstate0,qcap);
            break;

        default:
            return(E_ORDER);

        }
	vecN_StateStore(ckt->CKTstate0,ccap,nstateccap);
	}
        break;

    default:
        errMsg = TMALLOC(char, strlen(methodmsg) + 1);
        strcpy(errMsg,methodmsg);
        return(E_METHOD);
    }
    /* not used
    *ceq = ckt->CKTstate0[ccap] - ckt->CKTag[0] * ckt->CKTstate0[qcap];
    *geq = ckt->CKTag[0] * cap;
    */
    return(OK);
}
