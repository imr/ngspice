/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"

int
HFET2bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel;
    int i ;

    /*  loop through all the hfet2 models */
    for( ; model != NULL; model = model->HFET2nextModel ) {
	HFET2instance *here;

        /* loop through all the instances of the model */
        for (here = model->HFET2instances; here != NULL ;
	    here = here->HFET2nextInstance) {

		i = 0 ;
		if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2sourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0)) {
			while (here->HFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2drainPriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainPriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0)) {
			while (here->HFET2sourcePriHFET2ourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourcePriHFET2ourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2sourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0)) {
			while (here->HFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2gateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0)) {
			while (here->HFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2drainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2drainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2sourcePriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HFET2sourcePriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
HFET2bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel;
    int i ;

    /*  loop through all the hfet2 models */
    for( ; model != NULL; model = model->HFET2nextModel ) {
	HFET2instance *here;

        /* loop through all the instances of the model */
        for (here = model->HFET2instances; here != NULL ;
	    here = here->HFET2nextInstance) {

		i = 0 ;
		if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2sourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0)) {
			while (here->HFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2drainPriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainPriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0)) {
			while (here->HFET2sourcePriHFET2ourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourcePriHFET2ourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2sourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0)) {
			while (here->HFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0)) {
			while (here->HFET2gateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0)) {
			while (here->HFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0)) {
			while (here->HFET2drainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2drainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0)) {
			while (here->HFET2sourcePriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HFET2sourcePriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
HFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;
    int i ;

    /*  loop through all the HfetB models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            i = 0 ;
            if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                while (here->HFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                while (here->HFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                while (here->HFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                while (here->HFET2sourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0))
            {
                while (here->HFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0))
            {
                while (here->HFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                while (here->HFET2drainPriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainPriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0))
            {
                while (here->HFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0))
            {
                while (here->HFET2sourcePriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourcePriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                while (here->HFET2sourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0))
            {
                while (here->HFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0))
            {
                while (here->HFET2gateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0))
            {
                while (here->HFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                while (here->HFET2drainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2drainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                while (here->HFET2sourcePriHFET2ourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HFET2sourcePriHFET2ourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}
