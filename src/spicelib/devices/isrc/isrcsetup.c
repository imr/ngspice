/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"

#include "ngspice/CUSPICE/CUSPICE.h"

/* ARGSUSED */
int
ISRCsetup (SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *state)
{
    ISRCmodel *model = (ISRCmodel *)inModel ;
    ISRCinstance *here ;

    int i, j, k, status ;

    NG_IGNORE(matrix) ;
    NG_IGNORE(ckt) ;
    NG_IGNORE(state) ;

    /* Counting the instances */
    for ( ; model != NULL ; model = ISRCnextModel(model))
    {
        i = 0 ;

        for (here = ISRCinstances(model); here != NULL ; here = ISRCnextInstance(here))
        {
            i++ ;
        }

        /* How many instances we have */
        model->n_instances = i ;
    }

    /*  loop through all the current source models */
    for (model = (ISRCmodel *)inModel ; model != NULL ; model = ISRCnextModel(model))
    {
        model->offsetRHS = ckt->total_n_valuesRHS ;

        k = 0 ;

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ; here = ISRCnextInstance(here))
        {
            /* For the RHS */
            if (here->ISRCposNode != 0)
                k++ ;

            if (here->ISRCnegNode != 0)
                k++ ;
        }

        model->n_valuesRHS = model->n_instances;
        ckt->total_n_valuesRHS += model->n_valuesRHS ;

        model->n_PtrRHS = k ;
        ckt->total_n_PtrRHS += model->n_PtrRHS ;


        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances; j++)
            model->PositionVectorRHS [j] = model->offsetRHS + j ;
    }

    /*  loop through all the current source models */
    for (model = (ISRCmodel *)inModel ; model != NULL ; model = ISRCnextModel(model))
    {
        status = cuISRCsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }

    return (OK) ;
}
