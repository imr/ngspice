/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT model from the circuit and frees the storage
 * it was using. returns an error if the model has instances
 */

#include "ngspice.h"
#include "nbjtdefs.h"
#include "sperror.h"
#include "suffix.h"

int
NBJTmDelete(inModel, modname, kill)
  GENmodel **inModel;
  IFuid modname;
  GENmodel *kill;

{

  NBJTmodel **model = (NBJTmodel **) inModel;
  NBJTmodel *modfast = (NBJTmodel *) kill;
  NBJTmodel **oldmod;
  oldmod = model;
  for (; *model; model = &((*model)->NBJTnextModel)) {
    if ((*model)->NBJTmodName == modname ||
	(modfast && *model == modfast))
      goto delgot;
    oldmod = model;
  }
  return (E_NOMOD);

delgot:
  if ((*model)->NBJTinstances)
    return (E_NOTEMPTY);
  *oldmod = (*model)->NBJTnextModel;	/* cut deleted device out of list */
  FREE(*model);
  return (OK);

}
