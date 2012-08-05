/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"

int
CCVSbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel;
    int i ;

    /*  loop through all the ccvs models */
    for( ; model != NULL; model = model->CCVSnextModel ) {
	CCVSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CCVSinstances; here != NULL ;
	    here = here->CCVSnextInstance) {

		i = 0 ;
		if ((here-> CCVSposNode != 0) && (here-> CCVSbranch != 0)) {
			while (here->CCVSposIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CCVSnegNode != 0) && (here-> CCVSbranch != 0)) {
			while (here->CCVSnegIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVSnegNode != 0)) {
			while (here->CCVSibrNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVSposNode != 0)) {
			while (here->CCVSibrPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVScontBranch != 0)) {
			while (here->CCVSibrContBrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->CCVSibrContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
CCVSbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel;
    int i ;

    /*  loop through all the ccvs models */
    for( ; model != NULL; model = model->CCVSnextModel ) {
	CCVSinstance *here;

        /* loop through all the instances of the model */
        for (here = model->CCVSinstances; here != NULL ;
	    here = here->CCVSnextInstance) {

		i = 0 ;
		if ((here-> CCVSposNode != 0) && (here-> CCVSbranch != 0)) {
			while (here->CCVSposIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CCVSnegNode != 0) && (here-> CCVSbranch != 0)) {
			while (here->CCVSnegIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVSnegNode != 0)) {
			while (here->CCVSibrNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVSposNode != 0)) {
			while (here->CCVSibrPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> CCVSbranch != 0) && (here-> CCVScontBranch != 0)) {
			while (here->CCVSibrContBrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->CCVSibrContBrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
CCVSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel ;
    CCVSinstance *here ;
    int i ;

    /*  loop through all the CurrentControlledVoltageSource models */
    for ( ; model != NULL ; model = model->CCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CCVSinstances ; here != NULL ; here = here->CCVSnextInstance)
        {
            i = 0 ;
            if ((here->CCVSposNode != 0) && (here->CCVSbranch != 0))
            {
                while (here->CCVSposIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCVSposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CCVSnegNode != 0) && (here->CCVSbranch != 0))
            {
                while (here->CCVSnegIbrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCVSnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CCVSbranch != 0) && (here->CCVSnegNode != 0))
            {
                while (here->CCVSibrNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCVSibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CCVSbranch != 0) && (here->CCVSposNode != 0))
            {
                while (here->CCVSibrPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCVSibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->CCVSbranch != 0) && (here->CCVScontBranch != 0))
            {
                while (here->CCVSibrContBrptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->CCVSibrContBrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}