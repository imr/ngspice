/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "vcvsdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCVSfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = model->VCVSnextModel) {
        for (here = model->VCVSinstances; here != NULL;
                here = here->VCVSnextInstance) {
            if(here->VCVSname == name) {
                if(here->VCVSbranch == 0) {
                    error = CKTmkCur(ckt,&tmp,here->VCVSname,"branch");
                    if(error) return(error);
                    here->VCVSbranch = tmp->number;
                }
                return(here->VCVSbranch);
            }
        }
    }
    return(0);
}
