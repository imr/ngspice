/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 3 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvdel.c

 DATE : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"

int HSMHVdelete(
     GENmodel *inModel,
     IFuid name,
     GENinstance **inInst)
{
  HSMHVinstance **fast = (HSMHVinstance**)inInst;
  HSMHVmodel *model = (HSMHVmodel*)inModel;
  HSMHVinstance **prev = NULL;
  HSMHVinstance *here;

  for( ;model ;model = model->HSMHVnextModel ) {
    prev = &(model->HSMHVinstances);
    for ( here = *prev ;here ;here = *prev ) {
      if ( here->HSMHVname == name || (fast && here==*fast) ) {
	*prev= here->HSMHVnextInstance;
	FREE(here);
	return(OK);
      }
      prev = &(here->HSMHVnextInstance);
    }
  }
  return(E_NODEV);
}
