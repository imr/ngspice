/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT2 model from the circuit and frees the storage
 * it was using. returns an error if the model has instances
 */

#include "ngspice.h"
#include "nbjt2def.h"
#include "sperror.h"
#include "suffix.h"

int
NBJT2mDelete(inModel, modname, kill)
  GENmodel **inModel;
  IFuid modname;
  GENmodel *kill;

{

  NBJT2model **model = (NBJT2model **) inModel;
  NBJT2model *modfast = (NBJT2model *) kill;
  NBJT2model **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->NBJT2nextModel)) {
    if ((*model)->NBJT2modName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  if ((*model)->NBJT2instances)
    return (E_NOTEMPTY);
  *oldmod = (*model)->NBJT2nextModel;	/* cut deleted device out of list */
  FREE(*model);
  return (OK);

}
