/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* NIsenReinit(ckt)
 *  Perform reinitialization necessary for the numerical iteration
 *  package - the matrix has now been fully accessed once, so we know
 *  how big it is, so allocate RHS vector
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"


#define CKALLOC(ptr,size,type) if(( ckt->ptr =\
TMALLOC(type, size)) == NULL) return(E_NOMEM);

int
NIsenReinit(CKTcircuit *ckt)
{
    int size;
    int senparms;
    int i;

#ifdef SENSDEBUG
    printf("NIsenReinit \n");
    printf("senflag = %d\n",ckt->CKTsenInfo->SENinitflag);
    if(ckt->CKTniState & NIUNINITIALIZED) {
        printf("circuit uninitialized\n");
    }
    else{
        printf("circuit already initialized\n");
    }
#endif /* SENSDEBUG */
    size = SMPmatSize(ckt->CKTmatrix);
    if(ckt->CKTsenInfo->SENinitflag){
        if(!(ckt->CKTniState & NIUNINITIALIZED)) {
#ifdef SENSDEBUG
            printf("NIsenReinit1\n");
#endif /* SENSDEBUG */
            if(ckt->CKTrhsOp) FREE(ckt->CKTrhsOp);
            if(ckt->CKTsenRhs) FREE(ckt->CKTsenRhs);
            if(ckt->CKTseniRhs) FREE(ckt->CKTseniRhs);
        }
        senparms = ckt->CKTsenInfo->SENparms;
#ifdef SENSDEBUG
        printf("NIsenReinit2\n");
#endif /* SENSDEBUG */
        /*
                   CKALLOC(CKTsenInfo->SEN_parmVal,senparms+1,double);
            */
        ckt->CKTsenInfo->SENsize = size;
        CKALLOC(CKTrhsOp,size+1,double);
        CKALLOC(CKTsenRhs,size+1,double);
        CKALLOC(CKTseniRhs,size+1,double);
        CKALLOC(CKTsenInfo->SEN_Sap,size+1,double*);
        CKALLOC(CKTsenInfo->SEN_RHS,size+1,double*);
        CKALLOC(CKTsenInfo->SEN_iRHS,size+1,double*);
        for(i=0;i<=(size);i++){
            CKALLOC(CKTsenInfo->SEN_Sap[i],senparms+1,double);
            CKALLOC(CKTsenInfo->SEN_RHS[i],senparms+1,double);
            CKALLOC(CKTsenInfo->SEN_iRHS[i],senparms+1,double);
        }
#ifdef SENSDEBUG
        printf("NIsenReinit3\n");
#endif /* SENSDEBUG */
        ckt->CKTsenInfo->SENinitflag = OFF;
    }
    return(0);
}
