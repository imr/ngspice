/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTsetup(ckt)
     * this is a driver program to iterate through all the various
     * setup functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"



#define CKALLOC(var,size,type) \
    if(size && (!(var =(type *)MALLOC((size)*sizeof(type))))){\
            return(E_NOMEM);\
}

extern SPICEdev *DEVices[];

int
CKTsetup(CKTcircuit *ckt)
{
    int i;
    int error;

    SMPmatrix *matrix;
    ckt->CKTnumStates=0;

#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        if (error = CKTsenSetup(ckt)) return(error);
    }
#endif

    if (ckt->CKTisSetup)
	return E_NOCHANGE;

    CKTpartition(ckt);

    error = NIinit(ckt);
    if (error) return(error);
    ckt->CKTisSetup = 1;

    matrix = ckt->CKTmatrix;

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVsetup != NULL) && (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVsetup))(matrix,ckt->CKThead[i],ckt,
                    &ckt->CKTnumStates);
            if(error) return(error);
        }
    }
    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
        CKALLOC(ckt->CKTstates[i],ckt->CKTnumStates,double);
    }
#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        /* to allocate memory to sensitivity structures if 
         * it is not done before */

        error = NIsenReinit(ckt); 
        if(error) return(error);
    }
#endif
    if(ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt);
        if(error) return(error);
    }

    return(OK);
}

int
CKTunsetup(CKTcircuit *ckt)
{
    int i, error, e2;

    error = OK;
    if (!ckt->CKTisSetup)
	return OK;

    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
        tfree(ckt->CKTstates[i]);
    }

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVunsetup != NULL) && (ckt->CKThead[i] != NULL) ){
            e2 = (*((*DEVices[i]).DEVunsetup))(ckt->CKThead[i],ckt);
	    if (!error && e2)
		error = e2;
        }
    }
    ckt->CKTisSetup = 0;
    if(error) return(error);

    NIdestroy(ckt);
    /*
    if (ckt->CKTmatrix)
        SMPdestroy(ckt->CKTmatrix);
    ckt->CKTmatrix = NULL;
    */

    return OK;
}
