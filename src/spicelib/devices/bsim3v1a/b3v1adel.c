/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1adel.c
**********/
/*
 */

#include "ngspice.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3v1Adelete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
BSIM3v1Ainstance **fast = (BSIM3v1Ainstance**)inInst;
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance **prev = NULL;
BSIM3v1Ainstance *here;

    for (; model ; model = model->BSIM3v1AnextModel) 
    {    prev = &(model->BSIM3v1Ainstances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3v1Aname == name || (fast && here==*fast))
	      {   *prev= here->BSIM3v1AnextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3v1AnextInstance);
         }
    }
    return(E_NODEV);
}


