/***********************************************************************
 HiSIM v1.1.0
 File: hsm1dest.c of HiSIM v1.1.0

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
#include "suffix.h"

void HSM1destroy(GENmodel **inModel)
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

