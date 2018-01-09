/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VCVSfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = VCVSnextModel(model)) {
        for (here = VCVSinstances(model); here != NULL;
                here = VCVSnextInstance(here)) {
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
