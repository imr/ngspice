/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

/*ARGSUSED*/
int
MUTtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;
    double ind1, ind2;

#ifdef USE_CUSPICE
    int i, status ;
#endif

    NG_IGNORE(ckt);

    /*  loop through all the mutual inductor models */
    for( ; model != NULL; model = model->MUTnextModel ) {

#ifdef USE_CUSPICE
    i = 0 ;
#endif

        /* loop through all the instances of the model */
        for (here = model->MUTinstances; here != NULL ;
                here=here->MUTnextInstance) {

            /* Value Processing for mutual inductors */
	   
	    ind1 = here->MUTind1->INDinduct;
	    ind2 = here->MUTind2->INDinduct;
	    
	    /*           _______
	 * M = k * \/l1 * l2 
	 */
            here->MUTfactor = here->MUTcoupling * sqrt(ind1 * ind2); 

#ifdef USE_CUSPICE
            model->MUTparamCPU.MUTfactorArray[i] = here->MUTfactor ;
            model->MUTparamCPU.MUTflux1Array[i] = here->MUTind1->INDflux ;
            model->MUTparamCPU.MUTflux2Array[i] = here->MUTind2->INDflux ;
            model->MUTparamCPU.MUTbrEq1Array[i] = here->MUTind1->INDbrEq ;
            model->MUTparamCPU.MUTbrEq2Array[i] = here->MUTind2->INDbrEq ;

            i++ ;
#endif

	}

#ifdef USE_CUSPICE
        status = cuMUTtemp ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
#endif

    }
    return (OK) ;
}
