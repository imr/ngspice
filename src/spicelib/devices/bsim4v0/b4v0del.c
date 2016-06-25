/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM4v0delete(inModel,name,inInst)
GENmodel *inModel;
IFuid name;
GENinstance **inInst;
{
BSIM4v0instance **fast = (BSIM4v0instance**)inInst;
BSIM4v0model *model = (BSIM4v0model*)inModel;
BSIM4v0instance **prev = NULL;
BSIM4v0instance *here;

    for (; model ; model = model->BSIM4v0nextModel) 
    {    prev = &(model->BSIM4v0instances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->BSIM4v0name == name || (fast && here==*fast))
	      {   *prev= here->BSIM4v0nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM4v0nextInstance);
         }
    }
    return(E_NODEV);
}
