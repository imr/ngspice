/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
Imported into hfet2 model: Paolo Nenzi 2001 
*/ 

#include "ngspice.h"
#include "hfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int
HFET2delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    HFET2model *model = (HFET2model*)inModel;
    HFET2instance **fast = (HFET2instance**)inst;
    HFET2instance **prev = NULL;
    HFET2instance *here;

    for( ; model ; model = model->HFET2nextModel) {
        prev = &(model->HFET2instances);
        for(here = *prev; here ; here = *prev) {
            if(here->HFET2name == name || (fast && here==*fast) ) {
                *prev= here->HFET2nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->HFET2nextInstance);
        }
    }
    return(E_NODEV);
}
