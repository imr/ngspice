/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

        /* load the inductor structure with those pointers needed later 
         * for fast matrix loading 
         */

#include "ngspice.h"
#include "ifsim.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


#ifdef MUTUAL
/*ARGSUSED*/
int
MUTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;
    int ktype;
    int error;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->MUTnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->MUTinstances; here != NULL ;
                here=here->MUTnextInstance) {
            
            ktype = CKTtypelook("Inductor");
            if(ktype <= 0) {
                (*(SPfrontEnd->IFerror))(ERR_PANIC,
                        "mutual inductor, but inductors not available!",
                        (IFuid *)NULL);
                return(E_INTERN);
            }

            error = CKTfndDev((void *)ckt,&ktype,(void **)&(here->MUTind1),
                    here->MUTindName1, (void *)NULL,(char *)NULL);
            if(error && error!= E_NODEV && error != E_NOMOD) return(error);
            if(error) {
                IFuid namarray[2];
                namarray[0]=here->MUTname;
                namarray[1]=here->MUTindName1;
                (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "%s: coupling to non-existant inductor %s.",
                    namarray);
            }
            error = CKTfndDev((void *)ckt,&ktype,(void **)&(here->MUTind2),
                    here->MUTindName2,(void *)NULL,(char *)NULL);
            if(error && error!= E_NODEV && error != E_NOMOD) return(error);
            if(error) {
                IFuid namarray[2];
                namarray[0]=here->MUTname;
                namarray[1]=here->MUTindName2;
                (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "%s: coupling to non-existant inductor %s.",
                    namarray);
            }


/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(MUTbr1br2,MUTind1->INDbrEq,MUTind2->INDbrEq)
            TSTALLOC(MUTbr2br1,MUTind2->INDbrEq,MUTind1->INDbrEq)
        }
    }
    return(OK);
}
#endif /* MUTUAL */
