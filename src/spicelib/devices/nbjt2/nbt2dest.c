/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NBJT2s from the circuit and frees all storage
 * they were using.  The current implementation has memory leaks.
 */

#include "ngspice.h"
#include "nbjt2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "suffix.h"

void
NBJT2destroy(inModel)
  GENmodel **inModel;

{

  NBJT2model **model = (NBJT2model **) inModel;
  NBJT2model *mod, *nextMod;
  NBJT2instance *inst, *nextInst;


  for (mod = *model; mod;) {
    for (inst = mod->NBJT2instances; inst;) {
      TWOdestroy(inst->NBJT2pDevice);
      nextInst = inst->NBJT2nextInstance;
      FREE(inst);
      inst = nextInst;
    }
    nextMod = mod->NBJT2nextModel;
    FREE(mod);
    mod = nextMod;
  }
  *model = NULL;
}
