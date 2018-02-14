/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int
NUMOSpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
  register NUMOSmodel *model = (NUMOSmodel *) inModel;
  register NUMOSinstance *inst;
  struct mosAdmittances yAc;
  double startTime;

  NG_IGNORE(ckt);

  for (; model != NULL; model = NUMOSnextModel(model)) {
    FieldDepMobility = model->NUMOSmodels->MODLfieldDepMobility;
    TransDepMobility = model->NUMOSmodels->MODLtransDepMobility;
    SurfaceMobility = model->NUMOSmodels->MODLsurfaceMobility;
    Srh = model->NUMOSmodels->MODLsrh;
    Auger = model->NUMOSmodels->MODLauger;
    AvalancheGen = model->NUMOSmodels->MODLavalancheGen;
    OneCarrier = model->NUMOSmethods->METHoneCarrier;
    AcAnalysisMethod = model->NUMOSmethods->METHacAnalysisMethod;
    MobDeriv = model->NUMOSmethods->METHmobDeriv;
    TWOacDebug = model->NUMOSoutputs->OUTPacDebug;

    for (inst = NUMOSinstances(model); inst != NULL;
         inst = NUMOSnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMOSglobals));

      NUMOSys(inst->NUMOSpDevice, s, &yAc);

      *(inst->NUMOSdrainDrainPtr) += yAc.yIdVdb.real;
      *(inst->NUMOSdrainDrainPtr + 1) += yAc.yIdVdb.imag;
      *(inst->NUMOSdrainSourcePtr) += yAc.yIdVsb.real;
      *(inst->NUMOSdrainSourcePtr + 1) += yAc.yIdVsb.imag;
      *(inst->NUMOSdrainGatePtr) += yAc.yIdVgb.real;
      *(inst->NUMOSdrainGatePtr + 1) += yAc.yIdVgb.imag;
      *(inst->NUMOSdrainBulkPtr) -=
	  yAc.yIdVdb.real + yAc.yIdVsb.real + yAc.yIdVgb.real;
      *(inst->NUMOSdrainBulkPtr + 1) -=
	  yAc.yIdVdb.imag + yAc.yIdVsb.imag + yAc.yIdVgb.imag;

      *(inst->NUMOSsourceDrainPtr) += yAc.yIsVdb.real;
      *(inst->NUMOSsourceDrainPtr + 1) += yAc.yIsVdb.imag;
      *(inst->NUMOSsourceSourcePtr) += yAc.yIsVsb.real;
      *(inst->NUMOSsourceSourcePtr + 1) += yAc.yIsVsb.imag;
      *(inst->NUMOSsourceGatePtr) += yAc.yIsVgb.real;
      *(inst->NUMOSsourceGatePtr + 1) += yAc.yIsVgb.imag;
      *(inst->NUMOSsourceBulkPtr) -=
	  yAc.yIsVdb.real + yAc.yIsVsb.real + yAc.yIsVgb.real;
      *(inst->NUMOSsourceBulkPtr + 1) -=
	  yAc.yIsVdb.imag + yAc.yIsVsb.imag + yAc.yIsVgb.imag;

      *(inst->NUMOSgateDrainPtr) += yAc.yIgVdb.real;
      *(inst->NUMOSgateDrainPtr + 1) += yAc.yIgVdb.imag;
      *(inst->NUMOSgateSourcePtr) += yAc.yIgVsb.real;
      *(inst->NUMOSgateSourcePtr + 1) += yAc.yIgVsb.imag;
      *(inst->NUMOSgateGatePtr) += yAc.yIgVgb.real;
      *(inst->NUMOSgateGatePtr + 1) += yAc.yIgVgb.imag;
      *(inst->NUMOSgateBulkPtr) -=
	  yAc.yIgVdb.real + yAc.yIgVsb.real + yAc.yIgVgb.real;
      *(inst->NUMOSgateBulkPtr + 1) -=
	  yAc.yIgVdb.imag + yAc.yIgVsb.imag + yAc.yIgVgb.imag;

      *(inst->NUMOSbulkDrainPtr) -=
	  yAc.yIdVdb.real + yAc.yIsVdb.real + yAc.yIgVdb.real;
      *(inst->NUMOSbulkDrainPtr + 1) -=
	  yAc.yIdVdb.imag + yAc.yIsVdb.imag + yAc.yIgVdb.imag;
      *(inst->NUMOSbulkSourcePtr) -=
	  yAc.yIdVsb.real + yAc.yIsVsb.real + yAc.yIgVsb.real;
      *(inst->NUMOSbulkSourcePtr + 1) -=
	  yAc.yIdVsb.imag + yAc.yIsVsb.imag + yAc.yIgVsb.imag;
      *(inst->NUMOSbulkGatePtr) -=
	  yAc.yIdVgb.real + yAc.yIsVgb.real + yAc.yIgVgb.real;
      *(inst->NUMOSbulkGatePtr + 1) -=
	  yAc.yIdVgb.imag + yAc.yIsVgb.imag + yAc.yIgVgb.imag;
      *(inst->NUMOSbulkBulkPtr) +=
	  yAc.yIdVdb.real + yAc.yIdVsb.real + yAc.yIdVgb.real +
	  yAc.yIsVdb.real + yAc.yIsVsb.real + yAc.yIsVgb.real +
	  yAc.yIgVdb.real + yAc.yIgVsb.real + yAc.yIgVgb.real;
      *(inst->NUMOSbulkBulkPtr + 1) -=
	  yAc.yIdVdb.imag + yAc.yIdVsb.imag + yAc.yIdVgb.imag +
	  yAc.yIsVdb.imag + yAc.yIsVsb.imag + yAc.yIsVgb.imag +
	  yAc.yIgVdb.imag + yAc.yIgVsb.imag + yAc.yIgVgb.imag;

      inst->NUMOSpDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
