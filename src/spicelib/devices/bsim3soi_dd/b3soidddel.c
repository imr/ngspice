/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidddel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice.h"
#include "b3soidddef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
B3SOIDDdelete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
B3SOIDDinstance **fast = (B3SOIDDinstance**)inInst;
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance **prev = NULL;
B3SOIDDinstance *here;

    for (; model ; model = model->B3SOIDDnextModel) 
    {    prev = &(model->B3SOIDDinstances);
         for (here = *prev; here ; here = *prev) 
	 {    if (here->B3SOIDDname == name || (fast && here==*fast))
	      {   *prev= here->B3SOIDDnextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->B3SOIDDnextInstance);
         }
    }
    return(E_NODEV);
}


