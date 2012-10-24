/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2del.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"

int HSM2delete(
     GENmodel *inModel,
     IFuid name,
     GENinstance **inInst)
{
  HSM2instance **fast = (HSM2instance**)inInst;
  HSM2model *model = (HSM2model*)inModel;
  HSM2instance **prev = NULL;
  HSM2instance *here;

  for( ;model ;model = model->HSM2nextModel ) {
    prev = &(model->HSM2instances);
    for ( here = *prev ;here ;here = *prev ) {
      if ( here->HSM2name == name || (fast && here==*fast) ) {
	*prev= here->HSM2nextInstance;
	FREE(here);
	return(OK);
      }
      prev = &(here->HSM2nextInstance);
    }
  }
  return(E_NODEV);
}
