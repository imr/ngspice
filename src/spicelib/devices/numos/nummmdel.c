/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NUMOS model from the circuit and frees the storage
 * it was using. returns an error if the model has instances
 */

#include "ngspice.h"
#include "numosdef.h"
#include "sperror.h"
#include "suffix.h"

int
NUMOSmDelete(inModel, modname, kill)
  GENmodel **inModel;
  IFuid modname;
  GENmodel *kill;

{

  NUMOSmodel **model = (NUMOSmodel **) inModel;
  NUMOSmodel *modfast = (NUMOSmodel *) kill;
  NUMOSmodel **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->NUMOSnextModel)) {
    if ((*model)->NUMOSmodName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  if ((*model)->NUMOSinstances)
    return (E_NOTEMPTY);
  *oldmod = (*model)->NUMOSnextModel;	/* cut deleted device out of list */
  FREE(*model);
  return (OK);

}
