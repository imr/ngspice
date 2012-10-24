/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2mdel.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2mDelete(
     GENmodel **inModel,
     IFuid modname,
     GENmodel *kill)
{
  HSM2model **model = (HSM2model**)inModel;
  HSM2model *modfast = (HSM2model*)kill;
  HSM2instance *here;
  HSM2instance *prev = NULL;
  HSM2model **oldmod;

  oldmod = model;
  for ( ;*model ;model = &((*model)->HSM2nextModel) ) {
    if ( (*model)->HSM2modName == modname || 
	 (modfast && *model == modfast) ) goto delgot;
    oldmod = model;
  }
  return(E_NOMOD);

 delgot:
  *oldmod = (*model)->HSM2nextModel; /* cut deleted device out of list */
  for ( here = (*model)->HSM2instances ; 
	here ;here = here->HSM2nextInstance ) {
    if (prev) FREE(prev);
    prev = here;
  }
  if (prev) FREE(prev);
  FREE(*model);
  return(OK);
}

