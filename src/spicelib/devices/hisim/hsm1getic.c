/***********************************************************************
 HiSIM v1.1.0
 File: hsm1getic.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int HSM1getic(GENmodel *inModel, CKTcircuit *ckt)
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

