/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"

int
VCCSbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    int i ;

    /*  loop through all the vccs models */
    for( ; model != NULL; model = model->VCCSnextModel ) {
	VCCSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VCCSinstances; here != NULL ;
	    here = here->VCCSnextInstance) {

		i = 0 ;
		if ((here-> VCCSposNode != 0) && (here-> VCCScontPosNode != 0)) {
			while (here->VCCSposContPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCCSposContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCCSposNode != 0) && (here-> VCCScontNegNode != 0)) {
			while (here->VCCSposContNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCCSposContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCCSnegNode != 0) && (here-> VCCScontPosNode != 0)) {
			while (here->VCCSnegContPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCCSnegContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCCSnegNode != 0) && (here-> VCCScontNegNode != 0)) {
			while (here->VCCSnegContNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCCSnegContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
VCCSbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    int i ;

    /*  loop through all the vccs models */
    for( ; model != NULL; model = model->VCCSnextModel ) {
	VCCSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VCCSinstances; here != NULL ;
	    here = here->VCCSnextInstance) {

		i = 0 ;
		if ((here-> VCCSposNode != 0) && (here-> VCCScontPosNode != 0)) {
			while (here->VCCSposContPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCCSposContPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCCSposNode != 0) && (here-> VCCScontNegNode != 0)) {
			while (here->VCCSposContNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCCSposContNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCCSnegNode != 0) && (here-> VCCScontPosNode != 0)) {
			while (here->VCCSnegContPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCCSnegContPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCCSnegNode != 0) && (here-> VCCScontNegNode != 0)) {
			while (here->VCCSnegContNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCCSnegContNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
VCCSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;
    int i ;

    /*  loop through all the VoltageControlledCurrentSource models */
    for ( ; model != NULL ; model = model->VCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCCSinstances ; here != NULL ; here = here->VCCSnextInstance)
        {
            i = 0 ;
            if ((here->VCCSposNode != 0) && (here->VCCScontPosNode != 0))
            {
                while (here->VCCSposContPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCCSposContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCCSposNode != 0) && (here->VCCScontNegNode != 0))
            {
                while (here->VCCSposContNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCCSposContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCCSnegNode != 0) && (here->VCCScontPosNode != 0))
            {
                while (here->VCCSnegContPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCCSnegContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCCSnegNode != 0) && (here->VCCScontNegNode != 0))
            {
                while (here->VCCSnegContNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCCSnegContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}