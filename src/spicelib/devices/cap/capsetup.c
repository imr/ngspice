/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

/*ARGSUSED*/
int
CAPsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the capacitor structure with those pointers needed later
         * for fast matrix loading
         */

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

        /*Default Value Processing for Model Parameters */
        if (!model->CAPmCapGiven) {
            model->CAPmCap = 0.0;
        }
        if (!model->CAPcjswGiven){
             model->CAPcjsw = 0.0;
        }
        if (!model->CAPdefWidthGiven) {
            model->CAPdefWidth = 10.e-6;
        }
        if (!model->CAPdefLengthGiven) {
            model->CAPdefLength = 0.0;
        }
        if (!model->CAPnarrowGiven) {
            model->CAPnarrow = 0.0;
        }
        if (!model->CAPshortGiven) {
            model->CAPshort = 0.0;
        }
        if (!model->CAPdelGiven) {
            model->CAPdel = 0.0;
        }
        if (!model->CAPtc1Given) {
            model->CAPtempCoeff1 = 0.0;
        }
        if (!model->CAPtc2Given) {
            model->CAPtempCoeff2 = 0.0;
        }
        if (!model->CAPtnomGiven) {
            model->CAPtnom = ckt->CKTnomTemp;
        }
        if (!model->CAPdiGiven) {
            model->CAPdi = 0.0;
        }
        if (!model->CAPthickGiven) {
            model->CAPthick = 0.0;
        }
        if (!model->CAPbv_maxGiven) {
            model->CAPbv_max = 1e99;
        }

        if (!model->CAPcjGiven) {
            if((model->CAPthickGiven)
               && (model->CAPthick > 0.0)) {
               if (model->CAPdiGiven)
                 model->CAPcj = (model->CAPdi * CONSTepsZero) / model->CAPthick;
               else
                 model->CAPcj = CONSTepsSiO2 / model->CAPthick;
            } else {
               model->CAPcj = 0.0;
            }
        }

        if (model->CAPdelGiven) {
            if (!model->CAPnarrowGiven)
                model->CAPnarrow = 2 * model->CAPdel;
            if (!model->CAPshortGiven)
                model->CAPshort = 2 * model->CAPdel;
        }

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ;
                here=CAPnextInstance(here)) {

            /* Default Value Processing for Capacitor Instance */
            if (!here->CAPlengthGiven) {
                here->CAPlength = 0;
            }
            if (!here->CAPbv_maxGiven) {
                here->CAPbv_max = model->CAPbv_max;
            }

            here->CAPqcap = *states;
            *states += 2;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 2 * (ckt->CKTsenInfo->SENparms);
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(CAPposPosPtr,CAPposNode,CAPposNode);
            TSTALLOC(CAPnegNegPtr,CAPnegNode,CAPnegNode);
            TSTALLOC(CAPposNegPtr,CAPposNode,CAPnegNode);
            TSTALLOC(CAPnegPosPtr,CAPnegNode,CAPposNode);
        }
    }

#ifdef USE_CUSPICE
    int i, j, k, status ;

    /* Counting the instances */
    for (model = (CAPmodel *)inModel ; model != NULL ; model = CAPnextModel(model))
    {
        i = 0 ;

        for (here = CAPinstances(model); here != NULL ; here = CAPnextInstance(here))
        {
            i++ ;
        }

        /* How much instances we have */
        model->n_instances = i ;

        /* This model supports CUDA */
        model->has_cuda = 1 ;
    }

    /*  loop through all the capacitor models */
    for (model = (CAPmodel *)inModel ; model != NULL ; model = CAPnextModel(model))
    {
        model->offset = ckt->total_n_values ;
        model->offsetRHS = ckt->total_n_valuesRHS ;

        j = 0 ;
        k = 0 ;

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ; here = CAPnextInstance(here))
        {
            /* For the Matrix */
            if ((here->CAPposNode != 0) && (here->CAPposNode != 0))
                j++ ;

            if ((here->CAPnegNode != 0) && (here->CAPnegNode != 0))
                j++ ;

            if ((here->CAPposNode != 0) && (here->CAPnegNode != 0))
                j++ ;

            if ((here->CAPnegNode != 0) && (here->CAPposNode != 0))
                j++ ;

            /* For the RHS */
            if (here->CAPposNode != 0)
                k++ ;

            if (here->CAPnegNode != 0)
                k++ ;
        }

        model->n_values = model->n_instances;
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
            model->PositionVector [j] = model->offset + j ;

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

    /*  loop through all the capacitor models */
    for (model = (CAPmodel *)inModel ; model != NULL ; model = CAPnextModel(model))
    {
        status = cuCAPsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }
#endif

    return(OK);
}

