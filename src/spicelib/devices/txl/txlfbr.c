/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "txldefs.h"
#include "sperror.h"
#include "suffix.h"


int
TXLfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    TXLmodel *model = (TXLmodel *)inModel;
    TXLinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = model->TXLnextModel) {
        for (here = model->TXLinstances; here != NULL;
                here = here->TXLnextInstance) {
            if(here->TXLname == name) {
                if(here->TXLbranch == 0) {
                    error = CKTmkCur(ckt,&tmp,here->TXLname,"branch");
                    if(error) return(error);
                    here->TXLbranch = tmp->number;
                }
                return(here->TXLbranch);
            }
        }
    }
    return(0);
}
