/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1del.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "hsm1def.h"
#include "sperror.h"
#include "gendefs.h"
#include "suffix.h"

int 
HSM1delete(GENmodel *inModel, IFuid name, GENinstance **inInst)
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
