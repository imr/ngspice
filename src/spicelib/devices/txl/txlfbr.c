/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
TXLfindBr(CKTcircuit *ckt, GENmodel *inModel, IFuid name)
{
    TXLmodel *model = (TXLmodel *)inModel;
    TXLinstance *here;
    int error;
    CKTnode *tmp;

    for( ; model != NULL; model = TXLnextModel(model)) {
        for (here = TXLinstances(model); here != NULL;
                here = TXLnextInstance(here)) {
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
