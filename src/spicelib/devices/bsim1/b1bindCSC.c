/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"

int
B1bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel;
    int i ;

    /*  loop through all the b1 models */
    for( ; model != NULL; model = model->B1nextModel ) {
	B1instance *here;

        /* loop through all the instances of the model */
        for (here = model->B1instances; here != NULL ;
	    here = here->B1nextInstance) {

		i = 0 ;
		if ((here-> B1dNode != 0) && (here-> B1dNode != 0)) {
			while (here->B1DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1gNode != 0)) {
			while (here->B1GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNode != 0) && (here-> B1sNode != 0)) {
			while (here->B1SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1bNode != 0)) {
			while (here->B1BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1bNode != 0)) {
			while (here->B1GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1dNode != 0)) {
			while (here->B1DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1gNode != 0)) {
			while (here->B1BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1gNode != 0)) {
			while (here->B1DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1gNode != 0)) {
			while (here->B1SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1sNode != 0)) {
			while (here->B1SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1bNode != 0)) {
			while (here->B1DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1bNode != 0)) {
			while (here->B1SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
B1bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel;
    int i ;

    /*  loop through all the b1 models */
    for( ; model != NULL; model = model->B1nextModel ) {
	B1instance *here;

        /* loop through all the instances of the model */
        for (here = model->B1instances; here != NULL ;
	    here = here->B1nextInstance) {

		i = 0 ;
		if ((here-> B1dNode != 0) && (here-> B1dNode != 0)) {
			while (here->B1DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1gNode != 0)) {
			while (here->B1GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNode != 0) && (here-> B1sNode != 0)) {
			while (here->B1SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1bNode != 0)) {
			while (here->B1BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1bNode != 0)) {
			while (here->B1GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1gNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1sNodePrime != 0)) {
			while (here->B1DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1dNode != 0)) {
			while (here->B1DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1bNode != 0) && (here-> B1gNode != 0)) {
			while (here->B1BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1gNode != 0)) {
			while (here->B1DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1gNode != 0)) {
			while (here->B1SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1sNode != 0)) {
			while (here->B1SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1dNodePrime != 0) && (here-> B1bNode != 0)) {
			while (here->B1DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1bNode != 0)) {
			while (here->B1SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B1sNodePrime != 0) && (here-> B1dNodePrime != 0)) {
			while (here->B1SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
B1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;
    int i ;

    /*  loop through all the bsim1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            i = 0 ;
            if ((here->B1dNode != 0) && (here->B1dNode != 0))
            {
                while (here->B1DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1gNode != 0) && (here->B1gNode != 0))
            {
                while (here->B1GgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNode != 0) && (here->B1sNode != 0))
            {
                while (here->B1SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1bNode != 0) && (here->B1bNode != 0))
            {
                while (here->B1BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNodePrime != 0) && (here->B1dNodePrime != 0))
            {
                while (here->B1DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNodePrime != 0) && (here->B1sNodePrime != 0))
            {
                while (here->B1SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNode != 0) && (here->B1dNodePrime != 0))
            {
                while (here->B1DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1gNode != 0) && (here->B1bNode != 0))
            {
                while (here->B1GbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1gNode != 0) && (here->B1dNodePrime != 0))
            {
                while (here->B1GdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1gNode != 0) && (here->B1sNodePrime != 0))
            {
                while (here->B1GspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNode != 0) && (here->B1sNodePrime != 0))
            {
                while (here->B1SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1bNode != 0) && (here->B1dNodePrime != 0))
            {
                while (here->B1BdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1bNode != 0) && (here->B1sNodePrime != 0))
            {
                while (here->B1BspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNodePrime != 0) && (here->B1sNodePrime != 0))
            {
                while (here->B1DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNodePrime != 0) && (here->B1dNode != 0))
            {
                while (here->B1DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1bNode != 0) && (here->B1gNode != 0))
            {
                while (here->B1BgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNodePrime != 0) && (here->B1gNode != 0))
            {
                while (here->B1DPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNodePrime != 0) && (here->B1gNode != 0))
            {
                while (here->B1SPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNodePrime != 0) && (here->B1sNode != 0))
            {
                while (here->B1SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1dNodePrime != 0) && (here->B1bNode != 0))
            {
                while (here->B1DPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNodePrime != 0) && (here->B1bNode != 0))
            {
                while (here->B1SPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B1sNodePrime != 0) && (here->B1dNodePrime != 0))
            {
                while (here->B1SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}