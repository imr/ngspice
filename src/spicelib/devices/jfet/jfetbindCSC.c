/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"

int
JFETbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel;
    int i ;

    /*  loop through all the jfet models */
    for( ; model != NULL; model = model->JFETnextModel ) {
	JFETinstance *here;

        /* loop through all the instances of the model */
        for (here = model->JFETinstances; here != NULL ;
	    here = here->JFETnextInstance) {

		i = 0 ;
		if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourceNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0)) {
			while (here->JFETdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourceNode != 0)) {
			while (here->JFETsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0)) {
			while (here->JFETdrainDrainPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETgateGatePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0)) {
			while (here->JFETsourceSourcePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->JFETsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
JFETbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel;
    int i ;

    /*  loop through all the jfet models */
    for( ; model != NULL; model = model->JFETnextModel ) {
	JFETinstance *here;

        /* loop through all the instances of the model */
        for (here = model->JFETinstances; here != NULL ;
	    here = here->JFETnextInstance) {

		i = 0 ;
		if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourceNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETsourceSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourceSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0)) {
			while (here->JFETdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETdrainPrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainPrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourceNode != 0)) {
			while (here->JFETsourcePrimeSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourcePrimeSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETsourcePrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourcePrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0)) {
			while (here->JFETdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0)) {
			while (here->JFETgateGatePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETgateGatePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0)) {
			while (here->JFETsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainPrimeNode != 0)) {
			while (here->JFETdrainPrimeDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETdrainPrimeDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourcePrimeNode != 0)) {
			while (here->JFETsourcePrimeSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->JFETsourcePrimeSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
JFETbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;
    int i ;

    /*  loop through all the Jfet models */
    for ( ; model != NULL ; model = model->JFETnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFETinstances ; here != NULL ; here = here->JFETnextInstance)
        {
            i = 0 ;
            if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                while (here->JFETdrainDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETdrainDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                while (here->JFETgateDrainPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETgateDrainPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0))
            {
                while (here->JFETgateSourcePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETgateSourcePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0))
            {
                while (here->JFETdrainPrimeDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETdrainPrimeDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0))
            {
                while (here->JFETdrainPrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETdrainPrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0))
            {
                while (here->JFETsourcePrimeGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETsourcePrimeGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0))
            {
                while (here->JFETdrainDrainPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETdrainDrainPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0))
            {
                while (here->JFETgateGatePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETgateGatePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0))
            {
                while (here->JFETsourceSourcePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->JFETsourceSourcePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

        }
    }

    return (OK) ;
}