/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1getic of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1getic(GENmodel *inModel, CKTcircuit *ckt)
{
  HSM1model *model = (HSM1model*)inModel;
  HSM1instance *here;
  /*
   * grab initial conditions out of rhs array.   User specified, so use
   * external nodes to get values
   */

  for ( ;model ;model = model->HSM1nextModel ) {
    for ( here = model->HSM1instances; here ;here = here->HSM1nextInstance ) {
      
         
      if (here->HSM1owner != ARCHme)
              continue;
      
      if (!here->HSM1_icVBS_Given) {
	here->HSM1_icVBS = 
	  *(ckt->CKTrhs + here->HSM1bNode) - 
	  *(ckt->CKTrhs + here->HSM1sNode);
      }
      if (!here->HSM1_icVDS_Given) {
	here->HSM1_icVDS = 
	  *(ckt->CKTrhs + here->HSM1dNode) - 
	  *(ckt->CKTrhs + here->HSM1sNode);
      }
      if (!here->HSM1_icVGS_Given) {
	here->HSM1_icVGS = 
	  *(ckt->CKTrhs + here->HSM1gNode) - 
	  *(ckt->CKTrhs + here->HSM1sNode);
      }
    }
  }
  return(OK);
}

