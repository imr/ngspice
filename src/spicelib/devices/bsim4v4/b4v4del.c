/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM4v4delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM4v4instance **fast = (BSIM4v4instance**)inInst;
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance **prev = NULL;
BSIM4v4instance *here;

    for (; model ; model = model->BSIM4v4nextModel)
    {    prev = &(model->BSIM4v4instances);
         for (here = *prev; here ; here = *prev)
	 {    if (here->BSIM4v4name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4v4nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4v4nextInstance);
         }
    }
    return(E_NODEV);
}
