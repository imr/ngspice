/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2011 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 2 )
 Model Parameter VERSION : 1.22
 FILE : hsmhvmdel.c

 DATE : 2011.6.29

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHVmDelete(
     GENmodel **inModel,
     IFuid modname,
     GENmodel *kill)
{
  HSMHVmodel **model = (HSMHVmodel**)inModel;
  HSMHVmodel *modfast = (HSMHVmodel*)kill;
  HSMHVinstance *here;
  HSMHVinstance *prev = NULL;
  HSMHVmodel **oldmod;

  oldmod = model;
  for ( ;*model ;model = &((*model)->HSMHVnextModel) ) {
    if ( (*model)->HSMHVmodName == modname || 
	 (modfast && *model == modfast) ) goto delgot;
    oldmod = model;
  }
  return(E_NOMOD);

 delgot:
  *oldmod = (*model)->HSMHVnextModel; /* cut deleted device out of list */
  for ( here = (*model)->HSMHVinstances ; 
	here ;here = here->HSMHVnextInstance ) {
    if (prev) FREE(prev);
    prev = here;
  }
  if (prev) FREE(prev);
  FREE(*model);
  return(OK);
}

