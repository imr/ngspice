/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

int
RESsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit*ckt, int *state)
        /* load the resistor structure with those pointers needed later
         * for fast matrix loading
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;
#ifdef USE_CUSPICE
    int i, j, status ;
#endif
    NG_IGNORE(state);
    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        if(!model->RESbv_maxGiven)
            model->RESbv_max = 1e99;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

            if(!here->RESmGiven)
                here->RESm = 1.0;
            if(!here->RESbv_maxGiven)
                here->RESbv_max = model->RESbv_max;

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(RESposPosptr, RESposNode, RESposNode);
            TSTALLOC(RESnegNegptr, RESnegNode, RESnegNode);
            TSTALLOC(RESposNegptr, RESposNode, RESnegNode);
            TSTALLOC(RESnegPosptr, RESnegNode, RESposNode);
        }
    }

#ifdef USE_CUSPICE
//    int i, j, status ;

    /* Counting the instances */
    for (model = (RESmodel *)inModel ; model != NULL ; model = model->RESnextModel)
    {
        i = 0 ;

        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            i++ ;
        }

        /* How much instances we have */
        model->n_instances = i ;
    }

    /*  loop through all the resistor models */
    for (model = (RESmodel *)inModel ; model != NULL ; model = model->RESnextModel)
    {
        model->offset = ckt->total_n_values ;

        j = 0 ;

        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here->RESposNode != 0) && (here->RESposNode != 0))
                j++ ;

            if ((here->RESnegNode != 0) && (here->RESnegNode != 0))
                j++ ;

            if ((here->RESposNode != 0) && (here->RESnegNode != 0))
                j++ ;

            if ((here->RESnegNode != 0) && (here->RESposNode != 0))
                j++ ;
        }

        model->n_values = model->n_instances ;
        ckt->total_n_values += model->n_values ;

        model->n_Ptr = j ;
        ckt->total_n_Ptr += model->n_Ptr ;


        /* Position Vector assignment */
        model->PositionVector = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances ; j++)
            model->PositionVector [j] = model->offset + j ;
    }

    /*  loop through all the resistor models */
    for (model = (RESmodel *)inModel ; model != NULL ; model = model->RESnextModel)
    {
        status = cuRESsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }
#endif

    return (OK) ;
}
