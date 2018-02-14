/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VSRCfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = VSRCnextModel(model)) {
        for (here = VSRCinstances(model); here != NULL;
                here = VSRCnextInstance(here)) {
            if(here->VSRCname == name) {
                if(here->VSRCbranch == 0) {
                    error = CKTmkCur(ckt,&tmp,here->VSRCname,"branch");
                    if(error) return(error);
                    here->VSRCbranch = tmp->number;
                }
                return(here->VSRCbranch);
            }
        }
    }
    return(0);
}
