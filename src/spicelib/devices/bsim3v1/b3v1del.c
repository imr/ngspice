/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1del.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */


#include "ngspice.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM3v1delete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
BSIM3v1instance **fast = (BSIM3v1instance**)inInst;
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance **prev = NULL;
BSIM3v1instance *here;

    for (; model ; model = model->BSIM3v1nextModel) 
    {    prev = &(model->BSIM3v1instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM3v1name == name || (fast && here==*fast))
	      {   *prev= here->BSIM3v1nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3v1nextInstance);
         }
    }
    return(E_NODEV);
}


