/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifddel.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice.h"
#include "b3soifddef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
B3SOIFDdelete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
B3SOIFDinstance **fast = (B3SOIFDinstance**)inInst;
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance **prev = NULL;
B3SOIFDinstance *here;

    for (; model ; model = model->B3SOIFDnextModel) 
    {    prev = &(model->B3SOIFDinstances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->B3SOIFDname == name || (fast && here==*fast))
	      {   *prev= here->B3SOIFDnextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->B3SOIFDnextInstance);
         }
    }
    return(E_NODEV);
}


