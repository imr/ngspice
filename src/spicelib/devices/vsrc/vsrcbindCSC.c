/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"

int
VSRCbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->VSRCnextModel ) {
	VSRCinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
	    here = here->VSRCnextInstance) {

		i = 0 ;
		if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCposIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VSRCposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCnegIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VSRCnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0)) {
			while (here->VSRCibrNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VSRCibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0)) {
			while (here->VSRCibrPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VSRCibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
VSRCbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->VSRCnextModel ) {
	VSRCinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
	    here = here->VSRCnextInstance) {

		i = 0 ;
		if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCposIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VSRCposIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCnegIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VSRCnegIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0)) {
			while (here->VSRCibrNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VSRCibrNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0)) {
			while (here->VSRCibrPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VSRCibrPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
VSRCbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;
    int i ;

    /*  loop through all the source models */
    for ( ; model != NULL ; model = model->VSRCnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VSRCinstances ; here != NULL ; here = here->VSRCnextInstance)
        {
            i = 0 ;
            if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0))
            {
                while (here->VSRCposIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VSRCposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0))
            {
                while (here->VSRCnegIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VSRCnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0))
            {
                while (here->VSRCibrNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VSRCibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0))
            {
                while (here->VSRCibrPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VSRCibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}
