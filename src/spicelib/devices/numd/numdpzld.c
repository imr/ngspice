/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "numddefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "suffix.h"

/* External Declarations */
extern int ONEacDebug;

int
NUMDpzLoad(inModel, ckt, s)
  GENmodel *inModel;
  register CKTcircuit *ckt;
  SPcomplex *s;
{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  SPcomplex y;
  double startTime;

  /* loop through all the diode models */
  for (; model != NULL; model = model->NUMDnextModel) {
    FieldDepMobility = model->NUMDmodels->MODLfieldDepMobility;
    Srh = model->NUMDmodels->MODLsrh;
    Auger = model->NUMDmodels->MODLauger;
    AvalancheGen = model->NUMDmodels->MODLavalancheGen;
    AcAnalysisMethod = model->NUMDmethods->METHacAnalysisMethod;
    MobDeriv = model->NUMDmethods->METHmobDeriv;
    ONEacDebug = model->NUMDoutputs->OUTPacDebug;

    for (inst = model->NUMDinstances; inst != NULL;
	inst = inst->NUMDnextInstance) {
      if (inst->NUMDowner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();
      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMDglobals));

      NUMDys(inst->NUMDpDevice, s, &y);

      *(inst->NUMDposPosPtr) += y.real;
      *(inst->NUMDposPosPtr + 1) += y.imag;
      *(inst->NUMDnegNegPtr) += y.real;
      *(inst->NUMDnegNegPtr + 1) += y.imag;
      *(inst->NUMDnegPosPtr) -= y.real;
      *(inst->NUMDnegPosPtr + 1) -= y.imag;
      *(inst->NUMDposNegPtr) -= y.real;
      *(inst->NUMDposNegPtr + 1) -= y.imag;

      inst->NUMDpDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
