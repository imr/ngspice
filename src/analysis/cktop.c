/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"


int
CKTop(CKTcircuit *ckt, long int firstmode, long int continuemode, int iterlim)
{
    int converged;
    int i;

    ckt->CKTmode = firstmode;
    if(!ckt->CKTnoOpIter) {
        converged = NIiter(ckt,iterlim);
    } else {
        converged = 1;  /* the 'go directly to gmin stepping' option */
    }
    if(converged != 0) {
        /* no convergence on the first try, so we do something else */
        /* first, check if we should try gmin stepping */
        /* note that no path out of this code allows ckt->CKTdiagGmin to be 
         * anything but 0.000000000
         */
        if(ckt->CKTnumGminSteps >1) {
            ckt->CKTmode = firstmode;
            (*(SPfrontEnd->IFerror))(ERR_INFO,
                    "starting Gmin stepping",(IFuid *)NULL);
            ckt->CKTdiagGmin = ckt->CKTgmin;
            for(i=0;i<ckt->CKTnumGminSteps;i++) {
                ckt->CKTdiagGmin *= 10;
            }
            for(i=0;i<=ckt->CKTnumGminSteps;i++) {
                ckt->CKTnoncon =1;
                converged = NIiter(ckt,iterlim);
                if(converged != 0) {
                    ckt->CKTdiagGmin = 0;
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,
                            "Gmin step failed",(IFuid *)NULL);
                    break;
                }
                ckt->CKTdiagGmin /= 10;
                ckt->CKTmode=continuemode;
                (*(SPfrontEnd->IFerror))(ERR_INFO,
                        "One successful Gmin step",(IFuid *)NULL);
            }
            ckt->CKTdiagGmin = 0;
            converged = NIiter(ckt,iterlim);
            if(converged == 0) {
                (*(SPfrontEnd->IFerror))(ERR_INFO,
                        "Gmin stepping completed",(IFuid *)NULL);
                return(0);
            }
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "Gmin stepping failed",(IFuid *)NULL);

        }
        /* now, we'll try source stepping - we scale the sources
         * to 0, converge, then start stepping them up until they
         * are at their normal values
	 *
         * note that no path out of this code allows ckt->CKTsrcFact to be 
         * anything but 1.000000000
         */
        if(ckt->CKTnumSrcSteps >1) {
            ckt->CKTmode = firstmode;
            (*(SPfrontEnd->IFerror))(ERR_INFO,
                    "starting source stepping",(IFuid *)NULL);
            for(i=0;i<=ckt->CKTnumSrcSteps;i++) {
                ckt->CKTsrcFact = ((double)i)/((double)ckt->CKTnumSrcSteps);
                converged = NIiter(ckt,iterlim);
                ckt->CKTmode = continuemode;
                if(converged != 0) {
                    ckt->CKTsrcFact = 1;
                    ckt->CKTcurrentAnalysis = DOING_TRAN;
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,
                            "source stepping failed",(IFuid *)NULL);
                    return(converged);
                }
                (*(SPfrontEnd->IFerror))(ERR_INFO,
                        "One successful source step",(IFuid *)NULL);
            }
            (*(SPfrontEnd->IFerror))(ERR_INFO,
                    "Source stepping completed",(IFuid *)NULL);
            ckt->CKTsrcFact = 1;
            return(0);
        } else {
            return(converged);
        }
    } 
    return(0);
}

/* CKTconvTest(ckt)
 *    this is a driver program to iterate through all the various
 *    convTest functions provided for the circuit elements in the
 *    given circuit 
 */

int
CKTconvTest(CKTcircuit *ckt)
{
    extern SPICEdev *DEVices[];
    int i;
    int error = OK;
#ifdef PARALLEL_ARCH
    int ibuf[2];
    long type = MT_CONV, length = 2;
#endif /* PARALLEL_ARCH */

    for (i=0;i<DEVmaxnum;i++) {
        if (((*DEVices[i]).DEVconvTest != NULL) && (ckt->CKThead[i] != NULL)) {
            error = (*((*DEVices[i]).DEVconvTest))(ckt->CKThead[i],ckt);
        }
#ifdef PARALLEL_ARCH
	if (error || ckt->CKTnoncon) goto combine;
#else
	if (error) return(error);
	if (ckt->CKTnoncon) {
            /* printf("convTest: device %s failed\n",
                    (*DEVices[i]).DEVpublic.name); */
            return(OK);
        }
#endif /* PARALLEL_ARCH */
    }
#ifdef PARALLEL_ARCH
combine:
    /* See if any of the DEVconvTest functions bailed. If not, proceed. */
    ibuf[0] = error;
    ibuf[1] = ckt->CKTnoncon;
    IGOP_( &type, ibuf, &length, "+" );
    ckt->CKTnoncon = ibuf[1];
    if ( ibuf[0] != error ) {
	error = E_MULTIERR;
    }
    return (error);
#else
    return(OK);
#endif /* PARALLEL_ARCH */
}
