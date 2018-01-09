/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCVSfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = CCVSnextModel(model)) {
        for (here = CCVSinstances(model); here != NULL;
                here = CCVSnextInstance(here)) {
            if(here->CCVSname == name) {
                if(here->CCVSbranch == 0) {
                    error = CKTmkCur(ckt,&tmp, here->CCVSname,"branch");
                    if(error) return(error);
                    here->CCVSbranch = tmp->number;
                }
                return(here->CCVSbranch);
            }
        }
    }
    return(0);
}
