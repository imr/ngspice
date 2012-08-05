/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"

int
VCVSbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    int i ;

    /*  loop through all the vcvs models */
    for( ; model != NULL; model = model->VCVSnextModel ) {
	VCVSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
	    here = here->VCVSnextInstance) {

		i = 0 ;
		if ((here-> VCVSposNode != 0) && (here-> VCVSbranch != 0)) {
			while (here->VCVSposIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCVSnegNode != 0) && (here-> VCVSbranch != 0)) {
			while (here->VCVSnegIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVSnegNode != 0)) {
			while (here->VCVSibrNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVSposNode != 0)) {
			while (here->VCVSibrPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVScontPosNode != 0)) {
			while (here->VCVSibrContPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSibrContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVScontNegNode != 0)) {
			while (here->VCVSibrContNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VCVSibrContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
VCVSbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    int i ;

    /*  loop through all the vcvs models */
    for( ; model != NULL; model = model->VCVSnextModel ) {
	VCVSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
	    here = here->VCVSnextInstance) {

		i = 0 ;
		if ((here-> VCVSposNode != 0) && (here-> VCVSbranch != 0)) {
			while (here->VCVSposIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCVSnegNode != 0) && (here-> VCVSbranch != 0)) {
			while (here->VCVSnegIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVSnegNode != 0)) {
			while (here->VCVSibrNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVSposNode != 0)) {
			while (here->VCVSibrPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVScontPosNode != 0)) {
			while (here->VCVSibrContPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSibrContPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> VCVSbranch != 0) && (here-> VCVScontNegNode != 0)) {
			while (here->VCVSibrContNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VCVSibrContNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
VCVSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;
    int i ;

    /*  loop through all the VoltageControlledVoltageSource models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            i = 0 ;
            if ((here->VCVSposNode != 0) && (here->VCVSbranch != 0))
            {
                while (here->VCVSposIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCVSnegNode != 0) && (here->VCVSbranch != 0))
            {
                while (here->VCVSnegIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCVSbranch != 0) && (here->VCVSnegNode != 0))
            {
                while (here->VCVSibrNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCVSbranch != 0) && (here->VCVSposNode != 0))
            {
                while (here->VCVSibrPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCVSbranch != 0) && (here->VCVScontPosNode != 0))
            {
                while (here->VCVSibrContPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSibrContPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->VCVSbranch != 0) && (here->VCVScontNegNode != 0))
            {
                while (here->VCVSibrContNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->VCVSibrContNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}