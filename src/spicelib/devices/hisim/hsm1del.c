/***********************************************************************
 HiSIM v1.1.0
 File: hsm1del.c of HiSIM v1.1.0

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
#include "gendefs.h"
#include "suffix.h"

int HSM1delete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
  HSM1instance **fast = (HSM1instance**)inInst;
  HSM1model *model = (HSM1model*)inModel;
  HSM1instance **prev = NULL;
  HSM1instance *here;

  for( ;model ;model = model->HSM1nextModel ) {
    prev = &(model->HSM1instances);
    for ( here = *prev ;here ;here = *prev ) {
      if ( here->HSM1name == name || (fast && here==*fast) ) {
	*prev= here->HSM1nextInstance;
	FREE(here);
	return(OK);
      }
      prev = &(here->HSM1nextInstance);
    }
  }
  return(E_NODEV);
}
