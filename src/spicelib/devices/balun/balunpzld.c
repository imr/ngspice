/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "balundefs.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
BALUNpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    BALUNmodel *model = (BALUNmodel *)inModel;
    BALUNinstance *here;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = BALUNnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = BALUNinstances(model); here != NULL ;
                here=BALUNnextInstance(here)) {
            
            /* currents to and from nodes */
	    *(here->BALUNposIbrposPtr) -= 1.0;
	    *(here->BALUNnegIbrnegPtr) -= 1.0;
	    /* icm = ipos + ineg */
	    *(here->BALUNcmIbrposPtr) += 1.0;
	    *(here->BALUNcmIbrnegPtr) += 1.0;
	    /* idiff = (ipos - ineg)/2 */
	    *(here->BALUNdiffIbrposPtr) += 0.5;
	    *(here->BALUNdiffIbrnegPtr) -= 0.5;
	    	    	    
	    /* vd = vp - vn */
	    *(here->BALUNibrposDiffPtr) += 1.0 ;
	    *(here->BALUNibrposPosPtr) -= 1.0 ;
	    *(here->BALUNibrposNegPtr) += 1.0 ;
	    
	    /* vc = (vp + vn)/2 */
	    *(here->BALUNibrnegCmPtr) += 1.0 ;
	    *(here->BALUNibrnegPosPtr) -= 0.5 ;
	    *(here->BALUNibrnegNegPtr) -= 0.5 ;
        }
    }
    return(OK);
}
