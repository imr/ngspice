/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* NIintegrate(ckt,geq,ceq,cap,qcap)
 *  integrate the specified capacitor - method and order in the
 *  ckt structure, ccap follows qcap.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"

#define ccap qcap+1

int
NIintegrate(CKTcircuit *ckt, double *geq, double *ceq, double cap, int qcap)
{
    static char *ordmsg = "Illegal integration order";
    static char *methodmsg = "Unknown integration method";

    switch(ckt->CKTintegrateMethod) {

    case TRAPEZOIDAL:
        switch(ckt->CKTorder) {
        case 1:
            ckt->CKTstate0[ccap] = ckt->CKTag[0] * ckt->CKTstate0[qcap]
                    + ckt->CKTag[1] * ckt->CKTstate1[qcap];
            break;
        case 2:
            ckt->CKTstate0[ccap] = - ckt->CKTstate1[ccap] * ckt->CKTag[1] +
                    ckt->CKTag[0] *
                    ( ckt->CKTstate0[qcap] - ckt->CKTstate1[qcap] );
            break;
        default:
            errMsg = TMALLOC(char, strlen(ordmsg) + 1);
            strcpy(errMsg,ordmsg);
            return(E_ORDER);
        }
        break;
    case GEAR:
        ckt->CKTstate0[ccap]=0;
        switch(ckt->CKTorder) {

        case 6:
            ckt->CKTstate0[ccap] += ckt->CKTag[6]* ckt->CKTstate6[qcap];
            /* fall through */
        case 5:
            ckt->CKTstate0[ccap] += ckt->CKTag[5]* ckt->CKTstate5[qcap];
            /* fall through */
        case 4:
            ckt->CKTstate0[ccap] += ckt->CKTag[4]* ckt->CKTstate4[qcap];
            /* fall through */
        case 3:
            ckt->CKTstate0[ccap] += ckt->CKTag[3]* ckt->CKTstate3[qcap];
            /* fall through */
        case 2:
            ckt->CKTstate0[ccap] += ckt->CKTag[2]* ckt->CKTstate2[qcap];
            /* fall through */
        case 1:
            ckt->CKTstate0[ccap] += ckt->CKTag[1]* ckt->CKTstate1[qcap];
            ckt->CKTstate0[ccap] += ckt->CKTag[0]* ckt->CKTstate0[qcap];
            break;

        default:
            return(E_ORDER);

        }
        break;

    default:
        errMsg = TMALLOC(char, strlen(methodmsg) + 1);
        strcpy(errMsg,methodmsg);
        return(E_METHOD);
    }
    *ceq = ckt->CKTstate0[ccap] - ckt->CKTag[0] * ckt->CKTstate0[qcap];
    *geq = ckt->CKTag[0] * cap;
    return(OK);
}
