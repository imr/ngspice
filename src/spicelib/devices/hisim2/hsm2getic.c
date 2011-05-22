/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2011 Hiroshima University & STARC

 VERSION : HiSIM_2.5.1 
 FILE : hsm2getic.c

 date : 2011.04.07

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hsm2def.h"
#include "sperror.h"
#include "suffix.h"

int HSM2getic(inModel,ckt)
     GENmodel *inModel;
     CKTcircuit *ckt;
{
  HSM2model *model = (HSM2model*)inModel;
  HSM2instance *here;
  /*
   * grab initial conditions out of rhs array.   User specified, so use
   * external nodes to get values
   */

  for ( ;model ;model = model->HSM2nextModel ) {
    for ( here = model->HSM2instances; here ;here = here->HSM2nextInstance ) {
      if (!here->HSM2_icVBS_Given) {
	here->HSM2_icVBS = 
	  *(ckt->CKTrhs + here->HSM2bNode) - 
	  *(ckt->CKTrhs + here->HSM2sNode);
      }
      if (!here->HSM2_icVDS_Given) {
	here->HSM2_icVDS = 
	  *(ckt->CKTrhs + here->HSM2dNode) - 
	  *(ckt->CKTrhs + here->HSM2sNode);
      }
      if (!here->HSM2_icVGS_Given) {
	here->HSM2_icVGS = 
	  *(ckt->CKTrhs + here->HSM2gNode) - 
	  *(ckt->CKTrhs + here->HSM2sNode);
      }
    }
  }
  return(OK);
}

