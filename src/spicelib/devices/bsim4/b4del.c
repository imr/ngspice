/**** BSIM4.1.0, Released by Weidong Liu 10/1/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.1.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Weidong Liu, 10/11/2000.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4def.h"
#include "sperror.h"
#include "gendefs.h"


int
BSIM4delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM4instance **fast = (BSIM4instance**)inInst;
BSIM4model *model = (BSIM4model*)inModel;
BSIM4instance **prev = NULL;
BSIM4instance *here;

    for (; model ; model = model->BSIM4nextModel) 
    {    prev = &(model->BSIM4instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM4name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4nextInstance);
         }
    }
    return(E_NODEV);
}
