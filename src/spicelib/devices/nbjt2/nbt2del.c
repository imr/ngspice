/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT2 instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice.h"
#include "nbjt2def.h"
#include "sperror.h"
#include "suffix.h"

int
NBJT2delete(inModel, name, kill)
  GENmodel *inModel;
  IFuid name;
  GENinstance **kill;

{

  NBJT2model *model = (NBJT2model *) inModel;
  NBJT2instance **fast = (NBJT2instance **) kill;
  NBJT2instance **prev = NULL;
  NBJT2instance *inst;

  for (; model; model = model->NBJT2nextModel) {
    prev = &(model->NBJT2instances);
    for (inst = *prev; inst; inst = *prev) {
      if (inst->NBJT2name == name || (fast && inst == *fast)) {
	*prev = inst->NBJT2nextInstance;
	FREE(inst);
	return (OK);
      }
      prev = &(inst->NBJT2nextInstance);
    }
  }
  return (E_NODEV);
}
