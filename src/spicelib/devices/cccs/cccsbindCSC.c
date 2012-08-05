/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"

int
CCCSbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel;
    int i ;

    /*  loop through all the cccs models */
    for( ; model != NULL; model = model->CCCSnextModel ) {
	CCCSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
	    here = here->CCCSnextInstance) {

		i = 0 ;
		if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0)) {
			while (here->CCCSposContBrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCCSposContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0)) {
			while (here->CCCSnegContBrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCCSnegContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
CCCSbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel;
    int i ;

    /*  loop through all the cccs models */
    for( ; model != NULL; model = model->CCCSnextModel ) {
	CCCSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
	    here = here->CCCSnextInstance) {

		i = 0 ;
		if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0)) {
			while (here->CCCSposContBrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCCSposContBrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0)) {
			while (here->CCCSnegContBrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCCSnegContBrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
CCCSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel ;
    CCCSinstance *here ;
    int i ;

    /*  loop through all the CurrentControlledCurrentSource models */
    for ( ; model != NULL ; model = model->CCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CCCSinstances ; here != NULL ; here = here->CCCSnextInstance)
        {
            i = 0 ;
            if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0))
            {
                while (here->CCCSposContBrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCCSposContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0))
            {
                while (here->CCCSnegContBrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCCSnegContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}