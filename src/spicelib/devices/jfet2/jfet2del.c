/**********
Based on jfetdel.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/

#include "ngspice.h"
#include "jfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int
JFET2delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance **fast = (JFET2instance**)inst;
    JFET2instance **prev = NULL;
    JFET2instance *here;

    for( ; model ; model = model->JFET2nextModel) {
        prev = &(model->JFET2instances);
        for(here = *prev; here ; here = *prev) {
            if(here->JFET2name == name || (fast && here==*fast) ) {
                *prev= here->JFET2nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->JFET2nextInstance);
        }
    }
    return(E_NODEV);
}
