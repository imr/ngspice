/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2getic.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2getic(
     GENmodel *inModel,
     CKTcircuit *ckt)
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

