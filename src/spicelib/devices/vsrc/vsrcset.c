/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

/* ARGSUSED */
int
VSRCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *state)
        /* load the voltage source structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    CKTnode *tmp;
    int error;

    NG_IGNORE(state);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {
            
            if(here->VSRCposNode == here->VSRCnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted VSRC", here->VSRCname);
                return(E_UNSUPP);
            }

            if(here->VSRCbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->VSRCname,"branch");
                if(error) return(error);
                here->VSRCbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(VSRCposIbrPtr, VSRCposNode, VSRCbranch);
            TSTALLOC(VSRCnegIbrPtr, VSRCnegNode, VSRCbranch);
            TSTALLOC(VSRCibrNegPtr, VSRCbranch, VSRCnegNode);
            TSTALLOC(VSRCibrPosPtr, VSRCbranch, VSRCposNode);

#ifdef KLU
            here->VSRCibrIbrPtr = NULL ;
#endif

        }
    }

#ifdef USE_CUSPICE
    int i, j, k, status ;

    /* Counting the instances */
    for (model = (VSRCmodel *)inModel ; model != NULL ; model = VSRCnextModel(model))
    {
        i = 0 ;

        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            i++ ;
        }

        /* How much instances we have */
        model->n_instances = i ;
    }

    /*  loop through all the voltage source models */
    for (model = (VSRCmodel *)inModel ; model != NULL ; model = VSRCnextModel(model))
    {
        model->offset = ckt->total_n_values ;
        model->offsetRHS = ckt->total_n_valuesRHS ;

        j = 0 ;
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            /* For the Matrix */
            if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0))
                j++ ;

            if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0))
                j++ ;

            if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0))
                j++ ;

            if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0))
                j++ ;

            /* For the RHS */
            if (here->VSRCbranch != 0)
                k++ ;
        }

        model->n_values = model->n_instances ;
        ckt->total_n_values += model->n_values ;

        model->n_Ptr = j ;
        ckt->total_n_Ptr += model->n_Ptr ;

        model->n_valuesRHS = model->n_instances ;
        ckt->total_n_valuesRHS += model->n_valuesRHS ;

        model->n_PtrRHS = k ;
        ckt->total_n_PtrRHS += model->n_PtrRHS ;


        /* Position Vector assignment */
        model->PositionVector = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances ; j++)
            model->PositionVector [j] = model->offset + j ;

        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances ; j++)
            model->PositionVectorRHS [j] = model->offsetRHS + j ;
    }

    /*  loop through all the voltage source models */
    for (model = (VSRCmodel *)inModel ; model != NULL ; model = VSRCnextModel(model))
    {
        status = cuVSRCsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }
#endif

    return(OK);
}

int
VSRCunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model;
    VSRCinstance *here;

    for (model = (VSRCmodel *)inModel; model != NULL;
	    model = VSRCnextModel(model))
    {
        for (here = VSRCinstances(model); here != NULL;
                here=VSRCnextInstance(here))
	{
	    if (here->VSRCbranch > 0)
		CKTdltNNum(ckt, here->VSRCbranch);
            here->VSRCbranch = 0;
	}
    }
    return OK;
}
