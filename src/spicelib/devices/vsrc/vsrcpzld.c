/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "vsrcdefs.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
VSRCpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    for( ; model != NULL; model = model->VSRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
                here=here->VSRCnextInstance) {

            if (!(here->VSRCacGiven)) {
                /*a dc source*/
                /*the connecting nodes are shorted*/
                *(here->VSRCposIbrptr)  += 1.0 ;
                *(here->VSRCnegIbrptr)  += -1.0 ;
                *(here->VSRCibrPosptr)  += 1.0 ;
                *(here->VSRCibrNegptr)  += -1.0 ;
            } else {
                /*an ac source*/
                /*no effective contribution
                 *diagonal element made 1
                 */
                *(here->VSRCposIbrptr)  += 1.0 ;
                *(here->VSRCnegIbrptr)  += -1.0 ;
                *(here->VSRCibrIbrptr)  += 1.0 ;
            }
        }
    }
    return(OK);
}
