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

    int i, j, status ;

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
        model->gen.GENnInstances = i ;

        /* This model supports CUDA */
        model->gen.has_cuda = 1 ;
    }

    /*  loop through all the current source models */
    for (model = (ISRCmodel *)inModel ; model != NULL ; model = ISRCnextModel(model))
    {
        model->offsetRHS = ckt->total_n_valuesRHS ;

        j = 0 ;

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ; here = ISRCnextInstance(here))
        {
            /* For the RHS */
            if (here->ISRCposNode != 0)
                j++ ;

            if (here->ISRCnegNode != 0)
                j++ ;
        }

        model->n_valuesRHS = model->gen.GENnInstances;
        ckt->total_n_valuesRHS += model->n_valuesRHS ;

        model->n_PtrRHS = j ;
        ckt->total_n_PtrRHS += model->n_PtrRHS ;


        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC (int, model->gen.GENnInstances) ;

        for (j = 0 ; j < model->gen.GENnInstances; j++)
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
