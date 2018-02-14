/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "numd2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int
NUMD2pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  SPcomplex y;
  double startTime;

  NG_IGNORE(ckt);

  /* loop through all the diode models */
  for (; model != NULL; model = NUMD2nextModel(model)) {
    FieldDepMobility = model->NUMD2models->MODLfieldDepMobility;
    TransDepMobility = model->NUMD2models->MODLtransDepMobility;
    SurfaceMobility = model->NUMD2models->MODLsurfaceMobility;
    Srh = model->NUMD2models->MODLsrh;
    Auger = model->NUMD2models->MODLauger;
    AvalancheGen = model->NUMD2models->MODLavalancheGen;
    OneCarrier = model->NUMD2methods->METHoneCarrier;
    AcAnalysisMethod = model->NUMD2methods->METHacAnalysisMethod;
    MobDeriv = model->NUMD2methods->METHmobDeriv;
    TWOacDebug = model->NUMD2outputs->OUTPacDebug;

    for (inst = NUMD2instances(model); inst != NULL;
         inst = NUMD2nextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMD2globals));

      NUMD2ys(inst->NUMD2pDevice, s, &y);

      *(inst->NUMD2posPosPtr) += y.real;
      *(inst->NUMD2posPosPtr + 1) += y.imag;
      *(inst->NUMD2negNegPtr) += y.real;
      *(inst->NUMD2negNegPtr + 1) += y.imag;
      *(inst->NUMD2negPosPtr) -= y.real;
      *(inst->NUMD2negPosPtr + 1) -= y.imag;
      *(inst->NUMD2posNegPtr) -= y.real;
      *(inst->NUMD2posNegPtr + 1) -= y.imag;

      inst->NUMD2pDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
