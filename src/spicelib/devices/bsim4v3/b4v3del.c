/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3del.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v3def.h"
#include "sperror.h"
#include "gendefs.h"


int
BSIM4v3delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM4v3instance **fast = (BSIM4v3instance**)inInst;
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance **prev = NULL;
BSIM4v3instance *here;

    for (; model ; model = model->BSIM4v3nextModel) 
    {    prev = &(model->BSIM4v3instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM4v3name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4v3nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4v3nextInstance);
         }
    }
    return(E_NODEV);
}
