/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"

int
MESbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel;
    int i ;

    /*  loop through all the mes models */
    for( ; model != NULL; model = model->MESnextModel ) {
	MESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->MESinstances; here != NULL ;
	    here = here->MESnextInstance) {

		i = 0 ;
		if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourceNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0)) {
			while (here->MESdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESsourceNode != 0)) {
			while (here->MESsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0)) {
			while (here->MESdrainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESgateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0)) {
			while (here->MESsourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MESsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MESbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel;
    int i ;

    /*  loop through all the mes models */
    for( ; model != NULL; model = model->MESnextModel ) {
	MESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->MESinstances; here != NULL ;
	    here = here->MESnextInstance) {

		i = 0 ;
		if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourceNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0)) {
			while (here->MESdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESsourceNode != 0)) {
			while (here->MESsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0)) {
			while (here->MESdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESgateNode != 0) && (here->MESgateNode != 0)) {
			while (here->MESgateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESgateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0)) {
			while (here->MESsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESdrainPrimeNode != 0) && (here->MESdrainPrimeNode != 0)) {
			while (here->MESdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MESsourcePrimeNode != 0) && (here->MESsourcePrimeNode != 0)) {
			while (here->MESsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MESsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
MESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;
    int i ;

    /*  loop through all the mes models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            i = 0 ;
            if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                while (here->MESdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                while (here->MESgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0))
            {
                while (here->MESgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0))
            {
                while (here->MESdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0))
            {
                while (here->MESdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0))
            {
                while (here->MESsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0))
            {
                while (here->MESdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESgateNode != 0) && (here->MESgateNode != 0))
            {
                while (here->MESgateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0))
            {
                while (here->MESsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->MESsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

        }
    }

    return (OK) ;
}