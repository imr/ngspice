/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipddel.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "b3soipddef.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"


int
B3SOIPDdelete (inModel, name, inInst)
     GENmodel *inModel;
     IFuid name;
     GENinstance **inInst;
{
  B3SOIPDinstance **fast = (B3SOIPDinstance **) inInst;
  B3SOIPDmodel *model = (B3SOIPDmodel *) inModel;
  B3SOIPDinstance **prev = NULL;
  B3SOIPDinstance *here;

  for (; model; model = model->B3SOIPDnextModel)
    {
      prev = &(model->B3SOIPDinstances);
      for (here = *prev; here; here = *prev)
	{
	  if (here->B3SOIPDname == name || (fast && here == *fast))
	    {
	      *prev = here->B3SOIPDnextInstance;
	      FREE (here);
	      return (OK);
	    }
	  prev = &(here->B3SOIPDnextInstance);
	}
    }
  return (E_NODEV);
}
