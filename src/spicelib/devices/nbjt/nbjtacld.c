/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * Function to load the COMPLEX circuit matrix using the small signal
 * parameters saved during a previous DC operating point analysis.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/complex.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/numglobs.h"
#include "ngspice/suffix.h"

/* External Declarations */
extern int ONEacDebug;

int
NBJTacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  SPcomplex yIeVce, yIeVbe;
  SPcomplex yIcVce, yIcVbe;
  double startTime;

  for (; model != NULL; model = NBJTnextModel(model)) {
    FieldDepMobility = model->NBJTmodels->MODLfieldDepMobility;
    Srh = model->NBJTmodels->MODLsrh;
    Auger = model->NBJTmodels->MODLauger;
    AvalancheGen = model->NBJTmodels->MODLavalancheGen;
    AcAnalysisMethod = model->NBJTmethods->METHacAnalysisMethod;
    MobDeriv = model->NBJTmethods->METHmobDeriv;
    ONEacDebug = model->NBJToutputs->OUTPacDebug;

    for (inst = NBJTinstances(model); inst != NULL;
         inst = NBJTnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NBJTglobals));

      model->NBJTmethods->METHacAnalysisMethod =
	  NBJTadmittance(inst->NBJTpDevice, ckt->CKTomega,
	  &yIeVce, &yIcVce, &yIeVbe, &yIcVbe);

      *(inst->NBJTcolColPtr) += yIcVce.real;
      *(inst->NBJTcolColPtr + 1) += yIcVce.imag;
      *(inst->NBJTcolBasePtr) += yIcVbe.real;
      *(inst->NBJTcolBasePtr + 1) += yIcVbe.imag;
      *(inst->NBJTcolEmitPtr) -= yIcVbe.real + yIcVce.real;
      *(inst->NBJTcolEmitPtr + 1) -= yIcVbe.imag + yIcVce.imag;
      *(inst->NBJTbaseColPtr) -= yIcVce.real - yIeVce.real;
      *(inst->NBJTbaseColPtr + 1) -= yIcVce.imag - yIeVce.imag;
      *(inst->NBJTbaseBasePtr) -= yIcVbe.real - yIeVbe.real;
      *(inst->NBJTbaseBasePtr + 1) -= yIcVbe.imag - yIeVbe.imag;
      *(inst->NBJTbaseEmitPtr) += yIcVbe.real + yIcVce.real - yIeVbe.real - yIeVce.real;
      *(inst->NBJTbaseEmitPtr + 1) += yIcVbe.imag + yIcVce.imag - yIeVbe.imag - yIeVce.imag;
      *(inst->NBJTemitColPtr) -= yIeVce.real;
      *(inst->NBJTemitColPtr + 1) -= yIeVce.imag;
      *(inst->NBJTemitBasePtr) -= yIeVbe.real;
      *(inst->NBJTemitBasePtr + 1) -= yIeVbe.imag;
      *(inst->NBJTemitEmitPtr) += yIeVbe.real + yIeVce.real;
      *(inst->NBJTemitEmitPtr + 1) += yIeVbe.imag + yIeVce.imag;
      if (ckt->CKTomega != 0.0) {
	inst->NBJTc11 = yIcVce.imag / ckt->CKTomega;
	inst->NBJTc12 = yIcVbe.imag / ckt->CKTomega;
	inst->NBJTc21 = (yIeVce.imag - yIcVce.imag) / ckt->CKTomega;
	inst->NBJTc22 = (yIeVbe.imag - yIcVbe.imag) / ckt->CKTomega;
      } else {
	inst->NBJTc11 = 0.0;	/* XXX What else can be done?! */
	inst->NBJTc12 = 0.0;	/* XXX What else can be done?! */
	inst->NBJTc21 = 0.0;	/* XXX What else can be done?! */
	inst->NBJTc22 = 0.0;	/* XXX What else can be done?! */
      }
      inst->NBJTy11r = yIcVce.real;
      inst->NBJTy11i = yIcVce.imag;
      inst->NBJTy12r = yIcVbe.real;
      inst->NBJTy12i = yIcVbe.imag;
      inst->NBJTy21r = yIeVce.real - yIcVce.real;
      inst->NBJTy21i = yIeVce.imag - yIcVce.imag;
      inst->NBJTy22r = yIeVbe.real - yIcVbe.real;
      inst->NBJTy22i = yIeVbe.imag - yIcVbe.imag;
      inst->NBJTsmSigAvail = TRUE;
      inst->NBJTpDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
