/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2dest.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsm2def.h"
#include "ngspice/suffix.h"

void HSM2destroy(
     GENmodel **inModel)
{
  HSM2model **model = (HSM2model**)inModel;
  HSM2instance *here;
  HSM2instance *prev = NULL;
  HSM2model *mod = *model;
  HSM2model *oldmod = NULL;
  
  for ( ;mod ;mod = mod->HSM2nextModel ) {
    if (oldmod) FREE(oldmod);
    oldmod = mod;
    prev = (HSM2instance *)NULL;
    for ( here = mod->HSM2instances ;here ;here = here->HSM2nextInstance ) {
      if (prev) FREE(prev);
      prev = here;
    }
    if (prev) FREE(prev);
  }
  if (oldmod) FREE(oldmod);
  *model = NULL;
}

