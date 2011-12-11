/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numd2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMD2delete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
  NUMD2model *model = (NUMD2model *) inModel;
  NUMD2instance **fast = (NUMD2instance **) kill;
  NUMD2instance **prev = NULL;
  NUMD2instance *inst;

  for (; model; model = model->NUMD2nextModel) {
    prev = &(model->NUMD2instances);
    for (inst = *prev; inst; inst = *prev) {
      if (inst->NUMD2name == name || (fast && inst == *fast)) {
	*prev = inst->NUMD2nextInstance;
	FREE(inst);
	return (OK);
      }
      prev = &(inst->NUMD2nextInstance);
    }
  }
  return (E_NODEV);
}
