/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
File: b3del.c
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "bsim3def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM3instance **fast = (BSIM3instance**)inInst;
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance **prev = NULL;
BSIM3instance *here;

    for (; model ; model = model->BSIM3nextModel) 
    {    prev = &(model->BSIM3instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3name == name || (fast && here==*fast))
	      {   *prev= here->BSIM3nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3nextInstance);
         }
    }
    return(E_NODEV);
}


