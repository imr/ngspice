/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NUMOS instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice.h"
#include "numosdef.h"
#include "sperror.h"
#include "suffix.h"

int
NUMOSdelete(inModel, name, kill)
  GENmodel *inModel;
  IFuid name;
  GENinstance **kill;

{

  NUMOSmodel *model = (NUMOSmodel *) inModel;
  NUMOSinstance **fast = (NUMOSinstance **) kill;
  NUMOSinstance **prev = NULL;
  NUMOSinstance *inst;

  for (; model; model = model->NUMOSnextModel) {
    prev = &(model->NUMOSinstances);
    for (inst = *prev; inst; inst = *prev) {
      if (inst->NUMOSname == name || (fast && inst == *fast)) {
	*prev = inst->NUMOSnextInstance;
	FREE(inst);
	return (OK);
      }
      prev = &(inst->NUMOSnextInstance);
    }
  }
  return (E_NODEV);
}
