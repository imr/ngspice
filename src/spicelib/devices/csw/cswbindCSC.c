/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"

int
CSWbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel;
    int i ;

    /*  loop through all the csw models */
    for( ; model != NULL; model = model->CSWnextModel ) {
	CSWinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CSWinstances; here != NULL ;
	    here = here->CSWnextInstance) {

		i = 0 ;
		if ((here-> CSWposNode != 0) && (here-> CSWposNode != 0)) {
			while (here->CSWposPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CSWposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CSWposNode != 0) && (here-> CSWnegNode != 0)) {
			while (here->CSWposNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CSWposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CSWnegNode != 0) && (here-> CSWposNode != 0)) {
			while (here->CSWnegPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CSWnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CSWnegNode != 0) && (here-> CSWnegNode != 0)) {
			while (here->CSWnegNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CSWnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
CSWbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel;
    int i ;

    /*  loop through all the csw models */
    for( ; model != NULL; model = model->CSWnextModel ) {
	CSWinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CSWinstances; here != NULL ;
	    here = here->CSWnextInstance) {

		i = 0 ;
		if ((here-> CSWposNode != 0) && (here-> CSWposNode != 0)) {
			while (here->CSWposPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CSWposPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CSWposNode != 0) && (here-> CSWnegNode != 0)) {
			while (here->CSWposNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CSWposNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CSWnegNode != 0) && (here-> CSWposNode != 0)) {
			while (here->CSWnegPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CSWnegPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CSWnegNode != 0) && (here-> CSWnegNode != 0)) {
			while (here->CSWnegNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CSWnegNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
CSWbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;
    int i ;

    /*  loop through all the csw models */
    for ( ; model != NULL ; model = model->CSWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CSWinstances ; here != NULL ; here = here->CSWnextInstance)
        {
            i = 0 ;
            if ((here->CSWposNode != 0) && (here->CSWposNode != 0))
            {
                while (here->CSWposPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CSWposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CSWposNode != 0) && (here->CSWnegNode != 0))
            {
                while (here->CSWposNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CSWposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CSWnegNode != 0) && (here->CSWposNode != 0))
            {
                while (here->CSWnegPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CSWnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CSWnegNode != 0) && (here->CSWnegNode != 0))
            {
                while (here->CSWnegNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CSWnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}