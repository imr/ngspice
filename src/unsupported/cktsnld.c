/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* 
 * CKTsenLoad(ckt)
 * this is a driver program to iterate through all the various
 * sensitivity load functions provided for the circuit elements 
 * in the given circuit 
 */

#include "spice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"
#include "trandefs.h"
#include "suffix.h"


int
CKTsenLoad(ckt)
register CKTcircuit *ckt;
{
    extern SPICEdev *DEVices[];
    register int i;
    int size,row,col;
    int error;

    size = SMPmatSize(ckt->CKTmatrix);
#ifdef SENSDEBUG
    printf("CKTsenLoad\n");
#endif /* SENSDEBUG */

    if((ckt->CKTsenInfo->SENmode == DCSEN)||
            (ckt->CKTsenInfo->SENmode == TRANSEN)) {
        for (col=0;col<=ckt->CKTsenInfo->SENparms;col++) {
            for(row=0;row<=size;row++){
                *(ckt->CKTsenInfo->SEN_RHS[row] + col)= 0;
            }
        }
        for (i=0;i<DEVmaxnum;i++) {
            if ( ((*DEVices[i]).DEVsenLoad != NULL) &&
                    (ckt->CKThead[i] != NULL) ){
                error = (*((*DEVices[i]).DEVsenLoad))(ckt->CKThead[i],ckt);
                if(error) return(error);
            }
        }
    } else{ 
        for (col=0;col<=ckt->CKTsenInfo->SENparms;col++) {
            for(row=0;row<=size;row++){
                *(ckt->CKTsenInfo->SEN_RHS[row] + col)= 0;
                *(ckt->CKTsenInfo->SEN_iRHS[row] + col)= 0;
            }
        }
        for (i=0;i<DEVmaxnum;i++) {
            if ( ((*DEVices[i]).DEVsenAcLoad != NULL)
                    && (ckt->CKThead[i] != NULL) ){
                error = (*((*DEVices[i]).DEVsenAcLoad))(ckt->CKThead[i],ckt);
                if(error) return(error);
            }
        }
    }
    return(OK);
}
