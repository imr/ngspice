/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2del.c
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v2def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3V2delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM3V2instance **fast = (BSIM3V2instance**)inInst;
BSIM3V2model *model = (BSIM3V2model*)inModel;
BSIM3V2instance **prev = NULL;
BSIM3V2instance *here;

    for (; model ; model = model->BSIM3V2nextModel) 
    {    prev = &(model->BSIM3V2instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3V2name == name || (fast && here==*fast))
	      {   *prev= here->BSIM3V2nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3V2nextInstance);
         }
    }
    return(E_NODEV);
}


