/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NBJTs from the circuit and frees all storage they
 * were using.  The current implementation has memory leaks.
 */

#include "ngspice.h"
#include "nbjtdefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "suffix.h"

void
NBJTdestroy(inModel)
  GENmodel **inModel;

{

  NBJTmodel **model = (NBJTmodel **) inModel;
  NBJTmodel *mod, *nextMod;
  NBJTinstance *inst, *nextInst;


  for (mod = *model; mod;) {
    for (inst = mod->NBJTinstances; inst;) {
      ONEdestroy(inst->NBJTpDevice);
      nextInst = inst->NBJTnextInstance;
      FREE(inst);
      inst = nextInst;
    }
    nextMod = mod->NBJTnextModel;
    FREE(mod);
    mod = nextMod;
  }
  *model = NULL;
}
