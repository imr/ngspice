/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
BSIM4V4delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM4V4instance **fast = (BSIM4V4instance**)inInst;
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance **prev = NULL;
BSIM4V4instance *here;

    for (; model ; model = model->BSIM4V4nextModel) 
    {    prev = &(model->BSIM4V4instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM4V4name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4V4nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4V4nextInstance);
         }
    }
    return(E_NODEV);
}
