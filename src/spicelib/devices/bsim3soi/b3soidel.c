/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "b3soidef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
B3SOIdelete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
B3SOIinstance **fast = (B3SOIinstance**)inInst;
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance **prev = NULL;
B3SOIinstance *here;

    for (; model ; model = model->B3SOInextModel) 
    {    prev = &(model->B3SOIinstances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->B3SOIname == name || (fast && here==*fast))
	      {   *prev= here->B3SOInextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->B3SOInextInstance);
         }
    }
    return(E_NODEV);
}


