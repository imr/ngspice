/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * Function to load the COMPLEX circuit matrix using the small signal
 * parameters saved during a previous DC operating point analysis.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "numd2def.h"
#include "sperror.h"
#include "complex.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "cidersupt.h"
#include "numglobs.h"
#include "suffix.h"

/* External Declarations */
extern int TWOacDebug;

int
NUMD2acLoad(inModel, ckt)
  GENmodel *inModel;
  CKTcircuit *ckt;
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  SPcomplex y;
  double startTime;

  /* loop through all the diode models */
  for (; model != NULL; model = model->NUMD2nextModel) {
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

    for (inst = model->NUMD2instances; inst != NULL;
	inst = inst->NUMD2nextInstance) {
      if (inst->NUMD2owner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();
      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMD2globals));

      model->NUMD2methods->METHacAnalysisMethod =
	  NUMD2admittance(inst->NUMD2pDevice, ckt->CKTomega, &y);

      *(inst->NUMD2posPosPtr) += y.real;
      *(inst->NUMD2posPosPtr + 1) += y.imag;
      *(inst->NUMD2negNegPtr) += y.real;
      *(inst->NUMD2negNegPtr + 1) += y.imag;
      *(inst->NUMD2negPosPtr) -= y.real;
      *(inst->NUMD2negPosPtr + 1) -= y.imag;
      *(inst->NUMD2posNegPtr) -= y.real;
      *(inst->NUMD2posNegPtr + 1) -= y.imag;
      if (ckt->CKTomega != 0.0) {
	inst->NUMD2c11 = y.imag / ckt->CKTomega;
      } else {
	inst->NUMD2c11 = 0.0;	/* XXX What else can be done?! */
      }
      inst->NUMD2y11r = y.real;
      inst->NUMD2y11i = y.imag;
      inst->NUMD2smSigAvail = TRUE;
      inst->NUMD2pDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
