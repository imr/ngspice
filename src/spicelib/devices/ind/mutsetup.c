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


#define TSTALLOC(ptr, first, second)                                    \
    do {                                                                \
        if ((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL) { \
            return(E_NOMEM);                                            \
        }                                                               \
    } while(0)


int
MUTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    MUTmodel *model = (MUTmodel*) inModel;
    MUTinstance *here;

    NG_IGNORE(states);

    for (; model; model = MUTnextModel(model))
        for (here = MUTinstances(model); here; here = MUTnextInstance(here)) {

            int ktype = CKTtypelook("Inductor");

            if (ktype <= 0) {
                SPfrontEnd->IFerrorf (ERR_PANIC,
                                      "mutual inductor, but inductors not available!");
                return(E_INTERN);
            }

            if (!here->MUTind1)
                here->MUTind1 = (INDinstance *) CKTfndDev(ckt, here->MUTindName1);
            if (!here->MUTind1) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                                      "%s: coupling to non-existent inductor %s.",
                                      here->MUTname, here->MUTindName1);
                return(E_NOTFOUND); /* We have to leave, or TSTALLOC will segfault */
            }
            if (!here->MUTind2)
                here->MUTind2 = (INDinstance *) CKTfndDev(ckt, here->MUTindName2);
            if (!here->MUTind2) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                                      "%s: coupling to non-existent inductor %s.",
                                      here->MUTname, here->MUTindName2);
                return(E_NOTFOUND);
            }

            TSTALLOC(MUTbr1br2Ptr, MUTind1->INDbrEq, MUTind2->INDbrEq);
            TSTALLOC(MUTbr2br1Ptr, MUTind2->INDbrEq, MUTind1->INDbrEq);
        }

    return(OK);
}
