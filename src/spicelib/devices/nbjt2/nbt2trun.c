/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine performs truncation error calculations for NBJT2s in the
 * circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int
NBJT2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
  register NBJT2model *model = (NBJT2model *) inModel;
  register NBJT2instance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }

  for (; model != NULL; model = NBJT2nextModel(model)) {
    OneCarrier = model->NBJT2methods->METHoneCarrier;
    model->NBJT2pInfo->order = ckt->CKTorder;
    model->NBJT2pInfo->delta = deltaNorm;
    model->NBJT2pInfo->lteCoeff = computeLTECoeff(model->NBJT2pInfo);
    for (inst = NBJT2instances(model); inst != NULL;
         inst = NBJT2nextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      deltaNew = TWOtrunc(inst->NBJT2pDevice, model->NBJT2pInfo,
	  ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NBJT2pDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
