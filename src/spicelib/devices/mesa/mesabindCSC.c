/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"

int
MESAbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel;
    int i ;

    /*  loop through all the mesa models */
    for( ; model != NULL; model = model->MESAnextModel ) {
	MESAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->MESAinstances; here != NULL ;
	    here = here->MESAnextInstance) {

		i = 0 ;
		if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0)) {
			while (here->MESAdrainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0)) {
			while (here->MESAgateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0)) {
			while (here->MESAsourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0)) {
			while (here->MESAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0)) {
			while (here->MESAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0)) {
			while (here->MESAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MESAbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel;
    int i ;

    /*  loop through all the mesa models */
    for( ; model != NULL; model = model->MESAnextModel ) {
	MESAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->MESAinstances; here != NULL ;
	    here = here->MESAnextInstance) {

		i = 0 ;
		if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0)) {
			while (here->MESAdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0)) {
			while (here->MESAgateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0)) {
			while (here->MESAsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0)) {
			while (here->MESAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0)) {
			while (here->MESAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0)) {
			while (here->MESAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0)) {
			while (here->MESAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0)) {
			while (here->MESAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0)) {
			while (here->MESAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0)) {
			while (here->MESAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0)) {
			while (here->MESAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
MESAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;
    int i ;

    /*  loop through all the mesa models */
    for ( ; model != NULL ; model = model->MESAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESAinstances ; here != NULL ; here = here->MESAnextInstance)
        {
            i = 0 ;
            if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0))
            {
                while (here->MESAdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                while (here->MESAdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                while (here->MESAdrainPrmPrmDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrmPrmDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0))
            {
                while (here->MESAgateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAgatePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0))
            {
                while (here->MESAsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                while (here->MESAsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                while (here->MESAsourcePrmPrmSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrmPrmSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                while (here->MESAdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0))
            {
                while (here->MESAdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                while (here->MESAgatePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAdrainPrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                while (here->MESAgatePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAsourcePrimeGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrimeGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                while (here->MESAsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0))
            {
                while (here->MESAsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                while (here->MESAdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                while (here->MESAsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0))
            {
                while (here->MESAgatePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAgateGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgateGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                while (here->MESAsourcePrmPrmSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrmPrmSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                while (here->MESAsourcePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAsourcePrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAsourcePrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                while (here->MESAgatePrimeSourcePrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeSourcePrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                while (here->MESAdrainPrmPrmDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrmPrmDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                while (here->MESAdrainPrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                while (here->MESAdrainPrmPrmGatePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAdrainPrmPrmGatePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                while (here->MESAgatePrimeDrainPrmPrmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESAgatePrimeDrainPrmPrmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}