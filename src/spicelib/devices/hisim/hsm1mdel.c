/***********************************************************************
 HiSIM v1.1.0
 File: hsm1mdel.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int HSM1mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
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

