/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numd2def.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/suffix.h"
#include "ngspice/cidersupt.h"

int
NUMD2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }

  for (; model != NULL; model = NUMD2nextModel(model)) {
    OneCarrier = model->NUMD2methods->METHoneCarrier;
    model->NUMD2pInfo->order = ckt->CKTorder;
    model->NUMD2pInfo->delta = deltaNorm;
    model->NUMD2pInfo->lteCoeff = computeLTECoeff(model->NUMD2pInfo);
    for (inst = NUMD2instances(model); inst != NULL;
         inst = NUMD2nextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      deltaNew = TWOtrunc(inst->NUMD2pDevice, model->NUMD2pInfo,
	  ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NUMD2pDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
