/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "numddefs.h"
#include "sperror.h"
#include "suffix.h"

int
NUMDdelete(inModel, name, kill)
  GENmodel *inModel;
  IFuid name;
  GENinstance **kill;

{

  NUMDmodel *model = (NUMDmodel *) inModel;
  NUMDinstance **fast = (NUMDinstance **) kill;
  NUMDinstance **prev = NULL;
  NUMDinstance *inst;

  for (; model; model = model->NUMDnextModel) {
    prev = &(model->NUMDinstances);
    for (inst = *prev; inst; inst = *prev) {
      if (inst->NUMDname == name || (fast && inst == *fast)) {
	*prev = inst->NUMDnextInstance;
	FREE(inst);
	return (OK);
      }
      prev = &(inst->NUMDnextInstance);
    }
  }
  return (E_NODEV);
}
