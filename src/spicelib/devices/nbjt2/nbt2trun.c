/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine performs truncation error calculations for NBJT2s in the
 * circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "nbjt2def.h"
#include "sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "cidersupt.h"
#include "suffix.h"


int
NBJT2trunc(inModel, ckt, timeStep)
  GENmodel *inModel;
  register CKTcircuit *ckt;
  double *timeStep;

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

  for (; model != NULL; model = model->NBJT2nextModel) {
    OneCarrier = model->NBJT2methods->METHoneCarrier;
    model->NBJT2pInfo->order = ckt->CKTorder;
    model->NBJT2pInfo->delta = deltaNorm;
    model->NBJT2pInfo->lteCoeff = computeLTECoeff(model->NBJT2pInfo);
    for (inst = model->NBJT2instances; inst != NULL;
	inst = inst->NBJT2nextInstance) {
      if (inst->NBJT2owner != ARCHme) continue;

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
