/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
Imported into hfeta model: Paolo Nenzi 2001 
*/ 

#include "ngspice.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance **fast = (HFETAinstance**)inst;
    HFETAinstance **prev = NULL;
    HFETAinstance *here;

    for( ; model ; model = model->HFETAnextModel) {
        prev = &(model->HFETAinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->HFETAname == name || (fast && here==*fast) ) {
                *prev= here->HFETAnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->HFETAnextInstance);
        }
    }
    return(E_NODEV);
}
