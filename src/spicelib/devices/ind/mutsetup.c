/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

        /* load the inductor structure with those pointers needed later 
         * for fast matrix loading 
         */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


#ifdef MUTUAL
/*ARGSUSED*/
int
MUTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;
    int ktype;
    int error;

    NG_IGNORE(states);

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->MUTnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MUTinstances; here != NULL ;
                here=here->MUTnextInstance) {
            
            ktype = CKTtypelook("Inductor");
            if(ktype <= 0) {
                SPfrontEnd->IFerror (ERR_PANIC,
                        "mutual inductor, but inductors not available!",
                        NULL);
                return(E_INTERN);
            }

            // assert(third);
            here->MUTind1 = (INDinstance *) CKTfndDev(ckt, NULL, (GENinstance **) &(here->MUTind1), here->MUTindName1);
            error = here->MUTind1 ? OK : E_NODEV;
            if(error) {
                IFuid namarray[2];
                namarray[0]=here->MUTname;
                namarray[1]=here->MUTindName1;
                SPfrontEnd->IFerror (ERR_WARNING,
                    "%s: coupling to non-existant inductor %s.",
                    namarray);
            }
            // assert(third);
            here->MUTind2 = (INDinstance *) CKTfndDev(ckt, NULL, (GENinstance **) &(here->MUTind2), here->MUTindName2);
            error = here->MUTind2 ? OK : E_NODEV;
            if(error) {
                IFuid namarray[2];
                namarray[0]=here->MUTname;
                namarray[1]=here->MUTindName2;
                SPfrontEnd->IFerror (ERR_WARNING,
                    "%s: coupling to non-existant inductor %s.",
                    namarray);
            }


/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(MUTbr1br2,MUTind1->INDbrEq,MUTind2->INDbrEq);
            TSTALLOC(MUTbr2br1,MUTind2->INDbrEq,MUTind1->INDbrEq);
        }
    }
    return(OK);
}
#endif /* MUTUAL */
