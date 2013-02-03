/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2001 AlansFixes
**********/

/*
 * NIiter(ckt,maxIter)
 *
 *  This subroutine performs the actual numerical iteration.
 *  It uses the sparse matrix stored in the circuit struct
 *  along with the matrix loading program, the load data, the
 *  convergence test function, and the convergence parameters
 */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"


/* NIiter() - return value is non-zero for convergence failure */

int
NIiter(CKTcircuit *ckt, int maxIter)
{
    int iterno;
    int ipass;
    int error;
    int i,j; /* temporaries for finding error location */
    char *message;  /* temporary message buffer */
    double *temp;
    double startTime;
    static char *msg = "Too many iterations without convergence";

    CKTnode *node; /* current matrix entry */
    double diff, maxdiff, damp_factor, *OldCKTstate0=NULL;

    if ( maxIter < 100 ) maxIter = 100; /* some convergence issues that get resolved by increasing max iter */

    iterno=0;
    ipass=0;


    if( (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
        temp = ckt->CKTrhsOld;
        ckt->CKTrhsOld = ckt->CKTrhs;
        ckt->CKTrhs = temp;
        error = CKTload(ckt);
        if(error) {
            return(error);
        }
        return(OK);
    }
#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo) {
        error = NIsenReinit(ckt);
        if(error) return(error);
    }
#endif
    if(ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt);
        if(error) {
#ifdef STEPDEBUG
            printf("re-init returned error \n");
#endif
            return(error);
        }
    }

    /*    OldCKTstate0=TMALLOC(double, ckt->CKTnumStates + 1); */

    for(;;) {
        ckt->CKTnoncon=0;
#ifdef NEWPRED
        if(!(ckt->CKTmode & MODEINITPRED)) {
#else /* NEWPRED */
        if(1) { /* } */
#endif /* NEWPRED */
            error = CKTload(ckt);
            /*printf("loaded, noncon is %d\n",ckt->CKTnoncon);*/
            /*fflush(stdout);*/
            iterno++;
            if(error) {
                ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                printf("load returned error \n");
#endif
                FREE(OldCKTstate0);
                return(error);
            }
            /*printf("after loading, before solving\n");*/
            /*CKTdump(ckt);*/

            if(!(ckt->CKTniState & NIDIDPREORDER)) {
                error = SMPpreOrder(ckt->CKTmatrix);
                if(error) {
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("pre-order returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* badly formed matrix */
                }
                ckt->CKTniState |= NIDIDPREORDER;
            }
            if( (ckt->CKTmode & MODEINITJCT) ||
                    ( (ckt->CKTmode & MODEINITTRAN) && (iterno==1))) {
                ckt->CKTniState |= NISHOULDREORDER;
            }

            if(ckt->CKTniState & NISHOULDREORDER) {
                startTime = SPfrontEnd->IFseconds();
                error = SMPreorder(ckt->CKTmatrix,ckt->CKTpivotAbsTol,
                                   ckt->CKTpivotRelTol,ckt->CKTdiagGmin);
                ckt->CKTstat->STATreorderTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if(error) {
                    /* new feature - we can now find out something about what is
                     * wrong - so we ask for the troublesome entry
                     */
                    SMPgetError(ckt->CKTmatrix,&i,&j);
                    message = TMALLOC(char, 1000); /* should be enough */
                    (void)sprintf(message,
                                  "singular matrix:  check nodes %s and %s\n",
                                  NODENAME(ckt,i),NODENAME(ckt,j));
                    SPfrontEnd->IFerror (ERR_WARNING, message, NULL);
                    FREE(message);
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("reorder returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* can't handle these errors - pass up! */
                }
                ckt->CKTniState &= ~NISHOULDREORDER;
            } else {
                startTime = SPfrontEnd->IFseconds();
                error=SMPluFac(ckt->CKTmatrix,ckt->CKTpivotAbsTol,
                               ckt->CKTdiagGmin);
                ckt->CKTstat->STATdecompTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if(error) {
                    if( error == E_SINGULAR ) {
                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /*CKTload(ckt);*/
                    /*SMPprint(ckt->CKTmatrix,stdout);*/
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
            }
            /*moved it to here as if xspice is included then CKTload changes
              CKTnumStates the first time it is run */
            if(!OldCKTstate0)
                OldCKTstate0=TMALLOC(double, ckt->CKTnumStates + 1);
            for(i=0; i<ckt->CKTnumStates; i++) {
                OldCKTstate0[i] = ckt->CKTstate0[i];
            }

            startTime = SPfrontEnd->IFseconds();
            SMPsolve(ckt->CKTmatrix,ckt->CKTrhs,ckt->CKTrhsSpare);
            ckt->CKTstat->STATsolveTime += SPfrontEnd->IFseconds() -
                                           startTime;
#ifdef STEPDEBUG
            /*XXXX*/
            if (*ckt->CKTrhs != 0.0)
                printf("NIiter: CKTrhs[0] = %g\n", *ckt->CKTrhs);
            if (*ckt->CKTrhsSpare != 0.0)
                printf("NIiter: CKTrhsSpare[0] = %g\n", *ckt->CKTrhsSpare);
            if (*ckt->CKTrhsOld != 0.0)
                printf("NIiter: CKTrhsOld[0] = %g\n", *ckt->CKTrhsOld);
            /*XXXX*/
#endif
            *ckt->CKTrhs = 0;
            *ckt->CKTrhsSpare = 0;
            *ckt->CKTrhsOld = 0;

            if(iterno > maxIter) {
                /*fprintf(stderr,"too many iterations without convergence: %d iter's (max iter == %d)\n",
                iterno,maxIter);*/
                ckt->CKTstat->STATnumIter += iterno;
                FREE(errMsg);
                errMsg = TMALLOC(char, strlen(msg) + 1);
                strcpy(errMsg,msg);
#ifdef STEPDEBUG
                printf("iterlim exceeded \n");
#endif
                FREE(OldCKTstate0);
                return(E_ITERLIM);
            }
            if(ckt->CKTnoncon==0 && iterno!=1) {
                ckt->CKTnoncon = NIconvTest(ckt);
            } else {
                ckt->CKTnoncon = 1;
            }
#ifdef STEPDEBUG
            printf("noncon is %d\n",ckt->CKTnoncon);
#endif
        }

        if( (ckt->CKTnodeDamping!=0) && (ckt->CKTnoncon!=0) &&
                ((ckt->CKTmode & MODETRANOP) || (ckt->CKTmode & MODEDCOP)) &&
                (iterno>1) ) {
            maxdiff=0;
            for (node = ckt->CKTnodes->next; node; node = node->next) {
                if(node->type == SP_VOLTAGE) {
                    diff = ckt->CKTrhs [node->number] -
                           ckt->CKTrhsOld [node->number];
                    if (diff>maxdiff) maxdiff=diff;
                }
            }
            if (maxdiff>10) {
                damp_factor=10/maxdiff;
                if (damp_factor<0.1) damp_factor=0.1;
                for (node = ckt->CKTnodes->next; node; node = node->next) {
                    diff = (ckt->CKTrhs)[node->number] -
                           (ckt->CKTrhsOld)[node->number];
                    (ckt->CKTrhs)[node->number]=(ckt->CKTrhsOld)[node->number] +
                                                (damp_factor * diff);
                }
                for(i=0; i<ckt->CKTnumStates; i++) {
                    diff = ckt->CKTstate0[i] - OldCKTstate0[i];
                    ckt->CKTstate0[i] = OldCKTstate0[i] +
                                          (damp_factor * diff);
                }
            }
        }



        if(ckt->CKTmode & MODEINITFLOAT) {
            if ((ckt->CKTmode & MODEDC) &&
                    ( ckt->CKThadNodeset)  ) {
                if(ipass) {
                    ckt->CKTnoncon=ipass;
                }
                ipass=0;
            }
            if(ckt->CKTnoncon == 0) {
                ckt->CKTstat->STATnumIter += iterno;
                FREE(OldCKTstate0);
                return(OK);
            }
        } else if(ckt->CKTmode & MODEINITJCT) {
            ckt->CKTmode = (ckt->CKTmode&(~INITF))|MODEINITFIX;
            ckt->CKTniState |= NISHOULDREORDER;
        } else if (ckt->CKTmode & MODEINITFIX) {
            if(ckt->CKTnoncon==0) ckt->CKTmode =
                    (ckt->CKTmode&(~INITF))|MODEINITFLOAT;
            ipass=1;
        } else if (ckt->CKTmode & MODEINITSMSIG) {
            ckt->CKTmode = (ckt->CKTmode&(~INITF))|MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITTRAN) {
            if(iterno<=1) ckt->CKTniState |= NISHOULDREORDER;
            ckt->CKTmode = (ckt->CKTmode&(~INITF))|MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITPRED) {
            ckt->CKTmode = (ckt->CKTmode&(~INITF))|MODEINITFLOAT;
        } else {
            ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
            printf("bad initf state \n");
#endif
            FREE(OldCKTstate0);
            return(E_INTERN);
            /* impossible - no such INITF flag! */
        }

        /* build up the lvnim1 array from the lvn array */
        temp = ckt->CKTrhsOld;
        ckt->CKTrhsOld = ckt->CKTrhs;
        ckt->CKTrhs = temp;
        /*printf("after loading, after solving\n");*/
        /*CKTdump(ckt);*/
    }
    /*NOTREACHED*/
}
