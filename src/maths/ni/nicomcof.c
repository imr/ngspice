/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/cpextern.h"

/* xmu=0:    Backward Euler
 * xmu=0.5:  trapezoidal (standard)
 */


int
NIcomCof(CKTcircuit *ckt)
{

    double mat[8][8];   /* matrix to compute the gear coefficients in */
    int i,j,k;          /* generic loop indicies */
    double arg;
    double arg1;

    /*  this routine calculates the timestep-dependent terms used in the
     *  numerical integration.
     */ 

    /*  
     *  compute coefficients for particular integration method 
     */ 
    switch(ckt->CKTintegrateMethod) {

    case TRAPEZOIDAL:
        switch(ckt->CKTorder) {

        case 1:
            ckt->CKTag[0] = 1/ckt->CKTdelta;
            ckt->CKTag[1] = -1/ckt->CKTdelta;
            break;

        case 2:
            ckt->CKTag[0] = 1.0 / ckt->CKTdelta / (1.0 - ckt->CKTxmu);
            ckt->CKTag[1] = ckt->CKTxmu / (1.0 - ckt->CKTxmu);
            break;

        default:
            return(E_ORDER);
        }
        break;

    case GEAR:
        switch(ckt->CKTorder) {

        case 1:
        /*  ckt->CKTag[0] = 1/ckt->CKTdelta;
            ckt->CKTag[1] = -1/ckt->CKTdelta;
            break;*/

        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            memset(ckt->CKTag, 0, 7*sizeof(double));
            ckt->CKTag[1] = -1/ckt->CKTdelta;
            /* first, set up the matrix */
            arg=0;
            for(i=0;i<=ckt->CKTorder;i++) { mat[0][i]=1; }
            for(i=1;i<=ckt->CKTorder;i++) { mat[i][0]=0; }
            /* SPICE2 difference warning
             * the following block builds the corrector matrix
             * using (sum of h's)/h(final) instead of just (sum of h's)
             * because the h's are typically ~1e-10, so h^7 is an
             * underflow on many machines, but the ratio is ~1
             * and produces much better results
             */
            for(i=1;i<=ckt->CKTorder;i++) {
                arg += ckt->CKTdeltaOld[i-1];
                arg1 = 1;
                for(j=1;j<=ckt->CKTorder;j++) {
                    arg1 *= arg/ckt->CKTdelta;
                    mat[j][i]=arg1;
                }
            }
            /* lu decompose */
            /* weirdness warning! 
             * The following loop (and the first one after the forward
             * substitution comment) start at one instead of zero
             * because of a SPECIAL CASE - the first column is 1 0 0 ...
             * thus, the first iteration of both loops does nothing,
             * so it is skipped
             */
            for(i=1;i<=ckt->CKTorder;i++) {
                for(j=i+1;j<=ckt->CKTorder;j++) {
                    mat[j][i] /= mat[i][i];
                    for(k=i+1;k<=ckt->CKTorder;k++) {
                        mat[j][k] -= mat[j][i]*mat[i][k];
                    }
                }
            }
            /* forward substitution */
            for(i=1;i<=ckt->CKTorder;i++) {
                for(j=i+1;j<=ckt->CKTorder;j++) {
                    ckt->CKTag[j]=ckt->CKTag[j]-mat[j][i]*ckt->CKTag[i];
                }
            }
            /* backward substitution */
            ckt->CKTag[ckt->CKTorder] /= mat[ckt->CKTorder][ckt->CKTorder];
            for(i=ckt->CKTorder-1;i>=0;i--) {
                for(j=i+1;j<=ckt->CKTorder;j++) {
                    ckt->CKTag[i]=ckt->CKTag[i]-mat[i][j]*ckt->CKTag[j];
                }
                ckt->CKTag[i] /= mat[i][i];
            }
            break;
                

        default:
            return(E_ORDER);
        }
        break;

    default:
        return(E_METHOD);
    }

#ifdef PREDICTOR
    /* ok, have the coefficients for corrector, now for the predictor */

    switch(ckt->CKTintegrateMethod) {

    default:
        return(E_METHOD);

    case TRAPEZOIDAL:
        /*   ADAMS-BASHFORD PREDICTOR FOR TRAPEZOIDAL CORRECTOR
         *     MAY BE SUPPLEMENTED BY SECOND ORDER GEAR CORRECTOR
         *     AGP(1) STORES b0 AND AGP(2) STORES b1
         */
        arg = ckt->CKTdelta/(2*ckt->CKTdeltaOld[1]);
        ckt->CKTagp[0] = 1+arg;
        ckt->CKTagp[1] = -arg;
        break;

    case GEAR:

        /*
         *  CONSTRUCT GEAR PREDICTOR COEFICENT MATRIX
         *  MUST STILL ACCOUNT FOR ARRAY AGP()
         *  KEEP THE SAME NAME FOR GMAT
         */
        memset(ckt->CKTagp, 0, 7*sizeof(double));
        /*   SET UP RHS OF EQUATIONS */
        ckt->CKTagp[0]=1;
        for(i=0;i<=ckt->CKTorder;i++) {
            mat[0][i] = 1;
        }
        arg = 0;
        for(i=0;i<=ckt->CKTorder;i++){
            arg += ckt->CKTdeltaOld[i];
            arg1 = 1;
            for(j=1;j<=ckt->CKTorder;j++) {
                arg1 *= arg/ckt->CKTdelta;
                mat[j][i]=arg1;
            }
        }
        /*
         *  SOLVE FOR GEAR COEFFICIENTS AGP(*)
         */

        /*
         *  LU DECOMPOSITION
         */
        for(i=0;i<=ckt->CKTorder;i++) {
            for(j=i+1;j<=ckt->CKTorder;j++) {
                mat[j][i] /= mat[i][i];
                for(k=i+1;k<=ckt->CKTorder;k++) {
                    mat[j][k] -= mat[j][i]*mat[i][k];
                }
            }
        }
        /*
         *  FORWARD SUBSTITUTION
         */
        for(i=0;i<=ckt->CKTorder;i++) {
            for(j=i+1;j<=ckt->CKTorder;j++) {
                ckt->CKTagp[j] -= mat[j][i]*ckt->CKTagp[i];
            }
        }
        /*
         *  BACKWARD SUBSTITUTION
         */
        ckt->CKTagp[ckt->CKTorder] /= mat[ckt->CKTorder][ckt->CKTorder];
        for(i=ckt->CKTorder-1;i>=0;i--) {
            for(j=i+1;j<=ckt->CKTorder;j++) {
                ckt->CKTagp[i] -= mat[i][j]*ckt->CKTagp[j];
            }
            ckt->CKTagp[i] /= mat[i][i];
        }
        /*
         *  FINISHED
         */
        break;
    }
#endif /* PREDICTOR */
    return(OK);
}
