/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numddefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/suffix.h"
#include "ngspice/cidersupt.h"



int
NUMDtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }
  for (; model != NULL; model = NUMDnextModel(model)) {
    model->NUMDpInfo->order = ckt->CKTorder;
    model->NUMDpInfo->delta = deltaNorm;
    model->NUMDpInfo->lteCoeff = computeLTECoeff(model->NUMDpInfo);
    for (inst = NUMDinstances(model); inst != NULL;
         inst = NUMDnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      deltaNew = ONEtrunc(inst->NUMDpDevice, model->NUMDpInfo, ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NUMDpDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
