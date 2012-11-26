/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 3 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvdest.c

 DATE : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/suffix.h"

void HSMHVdestroy(
     GENmodel **inModel)
{
  HSMHVmodel **model = (HSMHVmodel**)inModel;
  HSMHVinstance *here;
  HSMHVinstance *prev = NULL;
  HSMHVmodel *mod = *model;
  HSMHVmodel *oldmod = NULL;
  
  for ( ;mod ;mod = mod->HSMHVnextModel ) {
    if (oldmod) FREE(oldmod);
    oldmod = mod;
    prev = (HSMHVinstance *)NULL;
    for ( here = mod->HSMHVinstances ;here ;here = here->HSMHVnextInstance ) {
      if (prev) FREE(prev);
      prev = here;
    }
    if (prev) FREE(prev);
  }
  if (oldmod) FREE(oldmod);
  *model = NULL;
}

