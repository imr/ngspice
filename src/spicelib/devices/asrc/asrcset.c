/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
ASRCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the voltage source structure with those 
     * pointers needed later for fast matrix loading 
         */

{
    ASRCinstance *here;
    ASRCmodel *model = (ASRCmodel*)inModel;
    int error, i, j;
    int v_first;
    CKTnode *tmp;

    /*  loop through all the user models*/
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
                here=here->ASRCnextInstance) {
            
            here->ASRCposptr = (double **)MALLOC(0);
            j=0; /*strchr of the array holding ptrs to SMP */
            v_first = 1;
            if( here->ASRCtype == ASRC_VOLTAGE){
                if(here->ASRCbranch==0) {
                    error = CKTmkCur(ckt,&tmp,here->ASRCname,"branch");
                    if(error) return(error);
                    here->ASRCbranch = tmp->number;
                }
            }


/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

#define MY_TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,((CKTnode*)(second))->number))\
        ==(double *)NULL){\
    return(E_NOMEM);\
}

            /* For each controlling variable set the entries
            in the vector of the positions of the SMP */
            if (!here->ASRCtree)
		return E_PARMVAL;
            for( i=0; i < here->ASRCtree->numVars; i++){
                switch(here->ASRCtree->varTypes[i]){
                case IF_INSTANCE:
                    here->ASRCcont_br = CKTfndBranch(ckt,
                            here->ASRCtree->vars[i].uValue);
                    if(here->ASRCcont_br == 0) {
                        IFuid namarray[2];
                        namarray[0] =  here->ASRCname;
                        namarray[1] = here->ASRCtree->vars[i].uValue;
                        (*(SPfrontEnd->IFerror))(ERR_FATAL,
                                "%s: unknown controlling source %s",namarray);
                        return(E_BADPARM);
                    }
                    if( here->ASRCtype == ASRC_VOLTAGE){
                        /* CCVS */
                        if(v_first){
                            here->ASRCposptr = (double **)
                            REALLOC(here->ASRCposptr, (sizeof(double *)*(j+5)));
                            TSTALLOC(ASRCposptr[j++],ASRCposNode,ASRCbranch);
                            TSTALLOC(ASRCposptr[j++],ASRCnegNode,ASRCbranch);
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCnegNode);
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCposNode);
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCcont_br);
                            v_first = 0;
                        } else{
                            here->ASRCposptr = (double **)
                            REALLOC(here->ASRCposptr, (sizeof(double *)*(j+1)));
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCcont_br);
                        }
                    } else if(here->ASRCtype == ASRC_CURRENT){
                        /* CCCS */
            here->ASRCposptr = (double **)
            REALLOC(here->ASRCposptr, (sizeof(double *) * (j+2)));
            TSTALLOC(ASRCposptr[j++],ASRCposNode,ASRCcont_br);
            TSTALLOC(ASRCposptr[j++],ASRCnegNode,ASRCcont_br);
                    } else{
                        return (E_BADPARM);
                    }
                    break;
                case IF_NODE:
                    if( here->ASRCtype == ASRC_VOLTAGE){
                        /* VCVS */
                        if(v_first){
                            here->ASRCposptr = (double **)
                        REALLOC(here->ASRCposptr, (sizeof(double *) * (j+5)));
                            TSTALLOC(ASRCposptr[j++],ASRCposNode,ASRCbranch);
                            TSTALLOC(ASRCposptr[j++],ASRCnegNode,ASRCbranch);
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCnegNode);
                            TSTALLOC(ASRCposptr[j++],ASRCbranch,ASRCposNode);
        MY_TSTALLOC(ASRCposptr[j++],ASRCbranch,here->ASRCtree->vars[i].nValue);
                            v_first = 0;
                        } else{
                            here->ASRCposptr = (double **)
                        REALLOC(here->ASRCposptr, (sizeof(double *) * (j+1)));
        MY_TSTALLOC(ASRCposptr[j++],ASRCbranch,here->ASRCtree->vars[i].nValue);
                        }
                    } else if(here->ASRCtype == ASRC_CURRENT){
                        /* VCCS */
                        here->ASRCposptr = (double **)
                    REALLOC(here->ASRCposptr, (sizeof(double *) * (j+2)));
        MY_TSTALLOC(ASRCposptr[j++],ASRCposNode,here->ASRCtree->vars[i].nValue);
        MY_TSTALLOC(ASRCposptr[j++],ASRCnegNode,here->ASRCtree->vars[i].nValue);
                    } else{
                        return (E_BADPARM);
                    }
                    break;
                default:
                    break;
                }
            }
        }
    }
    return(OK);
}

int
ASRCunsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    ASRCmodel *model;
    ASRCinstance *here;

    for (model = (ASRCmodel *)inModel; model != NULL;
	    model = model->ASRCnextModel)
    {
        for (here = model->ASRCinstances; here != NULL;
                here=here->ASRCnextInstance)
	{
	    if (here->ASRCbranch) {
		CKTdltNNum(ckt, here->ASRCbranch);
		here->ASRCbranch = 0;
	    }
	}
    }
    return OK;
}
