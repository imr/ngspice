/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2010 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 1 )
 Model Parameter VERSION : 1.21
 FILE : hsmhvmdel.c

 DATE : 2010.11.02

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice.h"
#include <stdio.h>
#include "hsmhvdef.h"
#include "sperror.h"
#include "suffix.h"

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

