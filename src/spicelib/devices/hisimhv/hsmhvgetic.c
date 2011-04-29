/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2010 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 1 )
 Model Parameter VERSION : 1.21
 FILE : hsmhvgetic.c

 DATE : 2010.11.02

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hsmhvdef.h"
#include "sperror.h"
#include "suffix.h"

int HSMHVgetic(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHVmodel *model = (HSMHVmodel*)inModel;
  HSMHVinstance *here;
  /*
   * grab initial conditions out of rhs array.   User specified, so use
   * external nodes to get values
   */

  for ( ;model ;model = model->HSMHVnextModel ) {
    for ( here = model->HSMHVinstances; here ;here = here->HSMHVnextInstance ) {
      if (!here->HSMHV_icVBS_Given) {
	here->HSMHV_icVBS = 
	  *(ckt->CKTrhs + here->HSMHVbNode) - 
	  *(ckt->CKTrhs + here->HSMHVsNode);
      }
      if (!here->HSMHV_icVDS_Given) {
	here->HSMHV_icVDS = 
	  *(ckt->CKTrhs + here->HSMHVdNode) - 
	  *(ckt->CKTrhs + here->HSMHVsNode);
      }
      if (!here->HSMHV_icVGS_Given) {
	here->HSMHV_icVGS = 
	  *(ckt->CKTrhs + here->HSMHVgNode) - 
	  *(ckt->CKTrhs + here->HSMHVsNode);
      }
    }
  }
  return(OK);
}

