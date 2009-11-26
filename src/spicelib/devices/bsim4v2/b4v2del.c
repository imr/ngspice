/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v2def.h"
#include "sperror.h"
#include "gendefs.h"


int
BSIM4v2delete(
GENmodel *inModel,
IFuid name,
GENinstance **inInst)
{
BSIM4v2instance **fast = (BSIM4v2instance**)inInst;
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance **prev = NULL;
BSIM4v2instance *here;

    for (; model ; model = model->BSIM4v2nextModel) 
    {    prev = &(model->BSIM4v2instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM4v2name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4v2nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4v2nextInstance);
         }
    }
    return(E_NODEV);
}
