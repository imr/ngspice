/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1dest.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "hsm1def.h"
#include "suffix.h"

void 
HSM1destroy(GENmodel **inModel)
{
  HSM1model **model = (HSM1model**)inModel;
  HSM1instance *here;
  HSM1instance *prev = NULL;
  HSM1model *mod = *model;
  HSM1model *oldmod = NULL;
  
  for ( ;mod ;mod = mod->HSM1nextModel ) {
    if (oldmod) FREE(oldmod);
    oldmod = mod;
    prev = (HSM1instance *)NULL;
    for ( here = mod->HSM1instances ;here ;here = here->HSM1nextInstance ) {
      if (prev) FREE(prev);
      prev = here;
    }
    if (prev) FREE(prev);
  }
  if (oldmod) FREE(oldmod);
  *model = NULL;
}

