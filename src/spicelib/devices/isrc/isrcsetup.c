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
    for ( ; model != NULL ; model = model->ISRCnextModel)
    {
        i = 0 ;

        for (here = model->ISRCinstances ; here != NULL ; here = here->ISRCnextInstance)
        {
            i++ ;
        }

        /* How many instances we have */
        model->n_instances = i ;
    }

    /*  loop through all the current source models */
    for (model = (ISRCmodel *)inModel ; model != NULL ; model = model->ISRCnextModel)
    {
        model->offsetRHS = ckt->total_n_valuesRHS ;

        k = 0 ;

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances ; here != NULL ; here = here->ISRCnextInstance)
        {
            /* For the RHS */
            if (here->ISRCposNode != 0)
                k++ ;

            if (here->ISRCnegNode != 0)
                k++ ;
        }

        model->n_valuesRHS = model->n_instances ;
        ckt->total_n_valuesRHS += model->n_valuesRHS ;

        model->n_PtrRHS = k ;
        ckt->total_n_PtrRHS += model->n_PtrRHS ;


        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC (int, model->n_instances) ;

        for (j = 0 ; j < model->n_instances ; j++)
            model->PositionVectorRHS [j] = model->offsetRHS + j ;
    }

    /*  loop through all the current source models */
    for (model = (ISRCmodel *)inModel ; model != NULL ; model = model->ISRCnextModel)
    {
        status = cuISRCsetup ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
    }

    return (OK) ;
}
