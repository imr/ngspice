/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1sdel.c
**********/
/*
 */

#include "ngspice.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3v1Sdelete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
BSIM3v1Sinstance **fast = (BSIM3v1Sinstance**)inInst;
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance **prev = NULL;
BSIM3v1Sinstance *here;

    for (; model ; model = model->BSIM3v1SnextModel) 
    {    prev = &(model->BSIM3v1Sinstances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3v1Sname == name || (fast && here==*fast))
	      {   *prev= here->BSIM3v1SnextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3v1SnextInstance);
         }
    }
    return(E_NODEV);
}


