/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice.h"
#include "nbjtdefs.h"
#include "sperror.h"
#include "suffix.h"

int
NBJTdelete(inModel, name, kill)
  GENmodel *inModel;
  IFuid name;
  GENinstance **kill;

{
  NBJTmodel *model = (NBJTmodel *) inModel;
  NBJTinstance **fast = (NBJTinstance **) kill;
  NBJTinstance **prev = NULL;
  NBJTinstance *inst;

  for (; model; model = model->NBJTnextModel) {
    prev = &(model->NBJTinstances);
    for (inst = *prev; inst; inst = *prev) {
      if (inst->NBJTname == name || (fast && inst == *fast)) {
	*prev = inst->NBJTnextInstance;
	FREE(inst);
	return (OK);
      }
      prev = &(inst->NBJTnextInstance);
    }
  }
  return (E_NODEV);
}
