/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NUMDs from the circuit and frees all storage they
 * were using.  The current implementation has memory leaks.
 */

#include "ngspice.h"
#include "numddefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "suffix.h"

void
NUMDdestroy(inModel)
  GENmodel **inModel;
{

  NUMDmodel **model = (NUMDmodel **) inModel;
  NUMDmodel *mod, *nextMod;
  NUMDinstance *inst, *nextInst;


  for (mod = *model; mod;) {
    for (inst = mod->NUMDinstances; inst;) {
      ONEdestroy(inst->NUMDpDevice);
      nextInst = inst->NUMDnextInstance;
      FREE(inst);
      inst = nextInst;
    }
    nextMod = mod->NUMDnextModel;
    FREE(mod);
    mod = nextMod;
  }
  *model = NULL;
}
