/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int
NBJT2pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
  register NBJT2model *model = (NBJT2model *) inModel;
  register NBJT2instance *inst;
  SPcomplex yIeVce, yIeVbe;
  SPcomplex yIcVce, yIcVbe;
  double startTime;

  NG_IGNORE(ckt);
 
  for (; model != NULL; model = NBJT2nextModel(model)) {
    FieldDepMobility = model->NBJT2models->MODLfieldDepMobility;
    TransDepMobility = model->NBJT2models->MODLtransDepMobility;
    SurfaceMobility = model->NBJT2models->MODLsurfaceMobility;
    Srh = model->NBJT2models->MODLsrh;
    Auger = model->NBJT2models->MODLauger;
    AvalancheGen = model->NBJT2models->MODLavalancheGen;
    OneCarrier = model->NBJT2methods->METHoneCarrier;
    AcAnalysisMethod = model->NBJT2methods->METHacAnalysisMethod;
    MobDeriv = model->NBJT2methods->METHmobDeriv;
    TWOacDebug = model->NBJT2outputs->OUTPacDebug;

    for (inst = NBJT2instances(model); inst != NULL;
         inst = NBJT2nextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();

      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NBJT2globals));

      NBJT2ys(inst->NBJT2pDevice, s,
	  &yIeVce, &yIcVce, &yIeVbe, &yIcVbe);

      *(inst->NBJT2colColPtr) += yIcVce.real;
      *(inst->NBJT2colColPtr + 1) += yIcVce.imag;
      *(inst->NBJT2colBasePtr) += yIcVbe.real;
      *(inst->NBJT2colBasePtr + 1) += yIcVbe.imag;
      *(inst->NBJT2colEmitPtr) -= yIcVbe.real + yIcVce.real;
      *(inst->NBJT2colEmitPtr + 1) -= yIcVbe.imag + yIcVce.imag;
      *(inst->NBJT2baseColPtr) -= yIcVce.real + yIeVce.real;
      *(inst->NBJT2baseColPtr + 1) -= yIcVce.imag + yIeVce.imag;
      *(inst->NBJT2baseBasePtr) -= yIcVbe.real + yIeVbe.real;
      *(inst->NBJT2baseBasePtr + 1) -= yIcVbe.imag + yIeVbe.imag;
      *(inst->NBJT2baseEmitPtr) += yIcVbe.real + yIcVce.real + yIeVbe.real + yIeVce.real;
      *(inst->NBJT2baseEmitPtr + 1) += yIcVbe.imag + yIcVce.imag + yIeVbe.imag + yIeVce.imag;
      *(inst->NBJT2emitColPtr) += yIeVce.real;
      *(inst->NBJT2emitColPtr + 1) += yIeVce.imag;
      *(inst->NBJT2emitBasePtr) += yIeVbe.real;
      *(inst->NBJT2emitBasePtr + 1) += yIeVbe.imag;
      *(inst->NBJT2emitEmitPtr) -= yIeVbe.real + yIeVce.real;
      *(inst->NBJT2emitEmitPtr + 1) -= yIeVbe.imag + yIeVce.imag;

      inst->NBJT2pDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
