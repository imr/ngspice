/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"

int
HFETAbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel;
    int i ;

    /*  loop through all the hfet models */
    for( ; model != NULL; model = model->HFETAnextModel ) {
	HFETAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->HFETAinstances; here != NULL ;
	    here = here->HFETAnextInstance) {

		i = 0 ;
		if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0)) {
			while (here->HFETAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0)) {
			while (here->HFETAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0)) {
			while (here->HFETAdrainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0)) {
			while (here->HFETAsourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0)) {
			while (here->HFETAgateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0)) {
			while (here->HFETAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFETAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
HFETAbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel;
    int i ;

    /*  loop through all the hfet models */
    for( ; model != NULL; model = model->HFETAnextModel ) {
	HFETAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->HFETAinstances; here != NULL ;
	    here = here->HFETAnextInstance) {

		i = 0 ;
		if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0)) {
			while (here->HFETAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0)) {
			while (here->HFETAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0)) {
			while (here->HFETAdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0)) {
			while (here->HFETAsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0)) {
			while (here->HFETAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0)) {
			while (here->HFETAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0)) {
			while (here->HFETAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0)) {
			while (here->HFETAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0)) {
			while (here->HFETAgateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0)) {
			while (here->HFETAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0)) {
			while (here->HFETAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFETAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
HFETAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;
    int i ;

    /*  loop through all the HfetA models */
    for ( ; model != NULL ; model = model->HFETAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFETAinstances ; here != NULL ; here = here->HFETAnextInstance)
        {
            i = 0 ;
            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                while (here->HFETAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                while (here->HFETAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                while (here->HFETAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                while (here->HFETAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0))
            {
                while (here->HFETAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                while (here->HFETAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0))
            {
                while (here->HFETAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                while (here->HFETAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0))
            {
                while (here->HFETAdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0))
            {
                while (here->HFETAsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                while (here->HFETAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                while (here->HFETAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                while (here->HFETAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                while (here->HFETAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                while (here->HFETAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                while (here->HFETAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                while (here->HFETAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                while (here->HFETAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                while (here->HFETAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                while (here->HFETAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0))
            {
                while (here->HFETAgateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                while (here->HFETAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0))
            {
                while (here->HFETAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFETAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}