/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
INDacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    double val;
    double m;
    INDinstance *here;

    for( ; model != NULL; model = INDnextModel(model)) {
        for( here = INDinstances(model);here != NULL; 
                here = INDnextInstance(here)) {
    
            m = (here->INDm);

            val = ckt->CKTomega * here->INDinduct / m;
	    
            *(here->INDposIbrPtr) +=  1;
            *(here->INDnegIbrPtr) -=  1;
            *(here->INDibrPosPtr) +=  1;
            *(here->INDibrNegPtr) -=  1;
            *(here->INDibrIbrPtr +1) -=  val;
        }
    }
    return(OK);

}
