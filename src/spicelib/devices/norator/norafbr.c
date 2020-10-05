/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "noradefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NORAfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    NORAmodel *model = (NORAmodel *)inModel;
    NORAinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = NORAnextModel(model)) {
        for (here = NORAinstances(model); here != NULL;
                here = NORAnextInstance(here)) {
            if(here->NORAname == name) {
                if(here->NORAbranch == 0) {
                    error = CKTmkCur(ckt,&tmp,here->NORAname,"branch");
                    if(error) return(error);
                    here->NORAbranch = tmp->number;
                }
                return(here->NORAbranch);
            }
        }
    }
    return(0);
}
