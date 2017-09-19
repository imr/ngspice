/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

int
INDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
   /* load the inductor structure with those pointers needed later 
   * for fast matrix loading 
   */
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the inductor models */
    for ( ; model != NULL ; model = INDnextModel(model))
    {
        /* Default Value Processing for Model Parameters */
        if (!model->INDmIndGiven) {
             model->INDmInd = 0.0;
        }
	if (!model->INDtnomGiven) {
             model->INDtnom = ckt->CKTnomTemp;
        }
	if (!model->INDtc1Given) {
             model->INDtempCoeff1 = 0.0;
        }
        if (!model->INDtc2Given) {
             model->INDtempCoeff2 = 0.0;
        }
        if (!model->INDcsectGiven){
             model->INDcsect = 0.0;
        }
        if (!model->INDlengthGiven) {
             model->INDlength = 0.0;
        }
        if (!model->INDmodNtGiven) {
             model->INDmodNt = 0.0;
        }
        if (!model->INDmuGiven) {
             model->INDmu = 0.0;
        }
          
	/* precompute specific inductance (one turn) */
	if((model->INDlengthGiven) 
              && (model->INDlength > 0.0)) {
                
		if (model->INDmuGiven)
                    model->INDspecInd = (model->INDmu * CONSTmuZero 
		     * model->INDcsect * model->INDcsect) / model->INDlength;   
		else
                   model->INDspecInd = (CONSTmuZero * model->INDcsect
		   * model->INDcsect ) / model->INDlength; 
	
	} else  {
	        model->INDspecInd = 0.0;
	}	
        
	if (!model->INDmIndGiven) 
            model->INDmInd = model->INDmodNt * model->INDmodNt * model->INDspecInd;
		   
            
        
	
        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            here->INDflux = *states;
            *states += 2 ;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 2 * (ckt->CKTsenInfo->SENparms);
            }

            if(here->INDbrEq == 0) {
                error = CKTmkCur(ckt,&tmp,here->INDname,"branch");
                if(error) return(error);
                here->INDbrEq = tmp->number;
            }

            here->system = NULL;
            here->system_next_ind = NULL;

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(INDposIbrPtr,INDposNode,INDbrEq);
            TSTALLOC(INDnegIbrPtr,INDnegNode,INDbrEq);
            TSTALLOC(INDibrNegPtr,INDbrEq,INDnegNode);
            TSTALLOC(INDibrPosPtr,INDbrEq,INDposNode);
            TSTALLOC(INDibrIbrPtr,INDbrEq,INDbrEq);
        }
    }

#ifdef USE_CUSPICE
    int i, j, k, status ;

    /* Counting the instances */
    for (model = (INDmodel *)inModel ; model != NULL ; model = INDnextModel(model))
    {
        i = 0 ;

        for (here = INDinstances(model); here != NULL ; here = INDnextInstance(here))
        {
            i++ ;
        }

        /* How much instances we have */
        model->n_instances = i ;

        /* This model supports CUDA */
        model->has_cuda = 1 ;
    }

    /*  loop through all the inductor models */
    for (model = (INDmodel *)inModel ; model != NULL ; model = INDnextModel(model))
    {
        model->offset = ckt->total_n_values ;
        model->offsetRHS = ckt->total_n_valuesRHS ;

        j = 0 ;
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ; here = INDnextInstance(here))
        {
            /* For the Matrix */
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
                j++ ;

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
                j++ ;

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
                j++ ;

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
                j++ ;

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
                j++ ;

            /* For the RHS */
            if (here->INDbrEq != 0)
                k++ ;
        }

        /* 2 Different Values for Every Instance */
        model->n_values = 2 * model->n_instances;
        ckt->total_n_values += model->n_values ;

        model->n_Ptr = j ;
        ckt->total_n_Ptr += model->n_Ptr ;

        model->n_valuesRHS = model->n_instances;
        ckt->total_n_valuesRHS += model->n_valuesRHS ;

        model->n_PtrRHS = k ;
        ckt->total_n_PtrRHS += model->n_PtrRHS ;


        /* Position Vector assignment */
        model->PositionVector = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances; j++)
        {
            /* 2 Different Values for Every Instance */
            model->PositionVector [j] = model->offset + 2 * j ;
        }

        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances; j++)
            model->PositionVectorRHS [j] = model->offsetRHS + j ;


        /* Position Vector for timeSteps */
        model->offset_timeSteps = ckt->total_n_timeSteps ;
        model->n_timeSteps = model->n_instances;
        ckt->total_n_timeSteps += model->n_timeSteps ;

        /* Position Vector assignment for timeSteps */
        model->PositionVector_timeSteps = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances; j++)
            model->PositionVector_timeSteps [j] = model->offset_timeSteps + j ;

    }

    /*  loop through all the inductor models */
    for (model = (INDmodel *)inModel ; model != NULL ; model = INDnextModel(model))
    {
        status = cuINDsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }
#endif

    return (OK) ;
}

int
INDunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model;
    INDinstance *here;

    for (model = (INDmodel *)inModel; model != NULL;
	    model = INDnextModel(model))
    {
        for (here = INDinstances(model); here != NULL;
                here=INDnextInstance(here))
	{
	    if (here->INDbrEq > 0)
		CKTdltNNum(ckt, here->INDbrEq);
            here->INDbrEq = 0;
	}
    }
    return OK;
}
