/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1dest.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice.h"
#include "bsim3v1def.h"
#include "suffix.h"

void
BSIM3v1destroy(GENmodel **inModel)
{
BSIM3v1model **model = (BSIM3v1model**)inModel;
BSIM3v1instance *here;
BSIM3v1instance *prev = NULL;
BSIM3v1model *mod = *model;
BSIM3v1model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3v1nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3v1instance *)NULL;
         for (here = mod->BSIM3v1instances; here; here = here->BSIM3v1nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



