/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0del.c
**********/
/*
 */

#include "ngspice.h"
#include "bsim3v0def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3v0delete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
BSIM3v0instance **fast = (BSIM3v0instance**)inInst;
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance **prev = NULL;
BSIM3v0instance *here;

    for (; model ; model = model->BSIM3v0nextModel) 
    {    prev = &(model->BSIM3v0instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3v0name == name || (fast && here==*fast))
	      {   *prev= here->BSIM3v0nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3v0nextInstance);
         }
    }
    return(E_NODEV);
}


