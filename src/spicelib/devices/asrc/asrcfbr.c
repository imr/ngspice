/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "ifsim.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
ASRCfindBr(ckt,inputModel,name)
    register CKTcircuit *ckt;
    GENmodel *inputModel;
    register IFuid name;
{
    register ASRCinstance *here;
    register ASRCmodel *model = (ASRCmodel*)inputModel;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = model->ASRCnextModel) {
        for (here = model->ASRCinstances; here != NULL;
                here = here->ASRCnextInstance) {
            if(here->ASRCname == name) {
                if(here->ASRCbranch == 0) {
                    error = CKTmkCur(ckt,&tmp, here->ASRCname,"branch");
                    if(error) return(error);
                    here->ASRCbranch = tmp->number;
                }
                return(here->ASRCbranch);
            }
        }
    }
    return(0);
}
