/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"

int
JFET2bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel;
    int i ;

    /*  loop through all the jfet2 models */
    for( ; model != NULL; model = model->JFET2nextModel ) {
	JFET2instance *here;

        /* loop through all the instances of the model */
        for (here = model->JFET2instances; here != NULL ;
	    here = here->JFET2nextInstance) {

		i = 0 ;
		if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourceNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2sourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0)) {
			while (here->JFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2drainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourceNode != 0)) {
			while (here->JFET2sourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2sourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0)) {
			while (here->JFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2gateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0)) {
			while (here->JFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2drainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2drainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2sourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFET2sourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
JFET2bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel;
    int i ;

    /*  loop through all the jfet2 models */
    for( ; model != NULL; model = model->JFET2nextModel ) {
	JFET2instance *here;

        /* loop through all the instances of the model */
        for (here = model->JFET2instances; here != NULL ;
	    here = here->JFET2nextInstance) {

		i = 0 ;
		if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourceNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2sourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0)) {
			while (here->JFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2drainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourceNode != 0)) {
			while (here->JFET2sourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2sourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0)) {
			while (here->JFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0)) {
			while (here->JFET2gateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0)) {
			while (here->JFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainPrimeNode != 0)) {
			while (here->JFET2drainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2drainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourcePrimeNode != 0)) {
			while (here->JFET2sourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFET2sourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
JFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;
    int i ;

    /*  loop through all the Jfet2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            i = 0 ;
            if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                while (here->JFET2drainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2drainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                while (here->JFET2gateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2gateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0))
            {
                while (here->JFET2gateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2gateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0))
            {
                while (here->JFET2drainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2drainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0))
            {
                while (here->JFET2drainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2drainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0))
            {
                while (here->JFET2sourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2sourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0))
            {
                while (here->JFET2drainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2drainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0))
            {
                while (here->JFET2gateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2gateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0))
            {
                while (here->JFET2sourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFET2sourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

        }
    }

    return (OK) ;
}