/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1mdel.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
  HSM1model **model = (HSM1model**)inModel;
  HSM1model *modfast = (HSM1model*)kill;
  HSM1instance *here;
  HSM1instance *prev = NULL;
  HSM1model **oldmod;

  oldmod = model;
  for ( ;*model ;model = &((*model)->HSM1nextModel) ) {
    if ( (*model)->HSM1modName == modname || 
	 (modfast && *model == modfast) ) goto delgot;
    oldmod = model;
  }
  return(E_NOMOD);

 delgot:
  *oldmod = (*model)->HSM1nextModel; /* cut deleted device out of list */
  for ( here = (*model)->HSM1instances ; 
	here ;here = here->HSM1nextInstance ) {
    if (prev) FREE(prev);
    prev = here;
  }
  if (prev) FREE(prev);
  FREE(*model);
  return(OK);
}

