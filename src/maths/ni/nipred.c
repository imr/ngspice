/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * NIpred(ckt)
 *
 *  This subroutine does node voltage prediction based on the
 *  integration method
 */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/smpdefs.h"

#ifdef PREDICTOR

int
NIpred(CKTcircuit * ckt)
{
    int i;
    int size;
    CKTnode *node;

    /* for our prediction, we have:
     *  ckt->CKTrhs[] is the current solution
     *  ckt->CKTsols[8][] is the set of previous solutions
     *
     * we want:
     *  ckt->CKTpred[] = ckt->CKTrhs = prediction based on proper number of
     *      previous time steps.
     */


    size = SMPmatSize(ckt->CKTmatrix);

    switch (ckt->CKTintegrateMethod) {

    case TRAPEZOIDAL: {
        double dd0, dd1, a, b;
        switch (ckt->CKTorder) {

        case 1:
            for (i = 0; i <= size; i++) {
                dd0 = (ckt->CKTsols[0][i] - ckt->CKTsols[1][i]) /
                    ckt->CKTdeltaOld[1];
                ckt->CKTpred[i] = ckt->CKTrhs[i] = ckt->CKTsols[0][i] +
                    ckt->CKTdeltaOld[0] * dd0;
            }
            break;

        case 2:
            for (i = 0; i <= size; i++) {
                b = - ckt->CKTdeltaOld[0] / (2*ckt->CKTdeltaOld[1]);
                a = 1 - b;
                dd0 = (ckt->CKTsols[0][i] - ckt->CKTsols[1][i]) /
                    ckt->CKTdeltaOld[1];
                dd1 = (ckt->CKTsols[1][i] - ckt->CKTsols[2][i]) /
                    ckt->CKTdeltaOld[2];
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTsols[0][i] +
                    (b * dd1 + a * dd0) * ckt->CKTdeltaOld[0];
            }
            break;

        default:
            return(E_ORDER);
        }
    }
        break;

    case GEAR:
        node = ckt->CKTnodes;
        switch (ckt->CKTorder) {

        case 1:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] ;
                node = node->next;
            }
            break;
        case 2:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] +
                    ckt->CKTagp[2] * ckt->CKTsols[2][i] ;
                node = node->next;
            }
            break;
        case 3:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] +
                    ckt->CKTagp[2] * ckt->CKTsols[2][i] +
                    ckt->CKTagp[3] * ckt->CKTsols[3][i] ;
            }
            break;
        case 4:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] +
                    ckt->CKTagp[2] * ckt->CKTsols[2][i] +
                    ckt->CKTagp[3] * ckt->CKTsols[3][i] +
                    ckt->CKTagp[4] * ckt->CKTsols[4][i] ;
            }
            break;
        case 5:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] +
                    ckt->CKTagp[2] * ckt->CKTsols[2][i] +
                    ckt->CKTagp[3] * ckt->CKTsols[3][i] +
                    ckt->CKTagp[4] * ckt->CKTsols[4][i] +
                    ckt->CKTagp[5] * ckt->CKTsols[5][i] ;
            }
            break;
        case 6:
            for (i = 0; i <= size; i++) {
                ckt->CKTpred[i] = ckt->CKTrhs[i] =
                    ckt->CKTagp[0] * ckt->CKTsols[0][i] +
                    ckt->CKTagp[1] * ckt->CKTsols[1][i] +
                    ckt->CKTagp[2] * ckt->CKTsols[2][i] +
                    ckt->CKTagp[3] * ckt->CKTsols[3][i] +
                    ckt->CKTagp[4] * ckt->CKTsols[4][i] +
                    ckt->CKTagp[5] * ckt->CKTsols[5][i] +
                    ckt->CKTagp[6] * ckt->CKTsols[6][i] ;
            }
            break;
        default:
            return(E_ORDER);
        }
        break;

    default:
        return(E_METHOD);
    }

    return(OK);
}

#else
int Dummy_Symbol;
#endif /* PREDICTOR */
