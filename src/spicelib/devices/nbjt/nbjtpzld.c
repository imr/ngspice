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
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/sperror.h"
#include "ngspice/complex.h"
#include "ngspice/suffix.h"

/* External Declarations */
extern int ONEacDebug;

int
NBJTpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  SPcomplex yIeVce, yIeVbe;
  SPcomplex yIcVce, yIcVbe;
  double startTime;

  NG_IGNORE(ckt);

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

      NBJTys(inst->NBJTpDevice, s,
	  &yIeVce, &yIcVce, &yIeVbe, &yIcVbe);

      if (ONEacDebug) {
	fprintf(stdout, "BJT admittances: %s:%s at s = % .5g, % .5g\n",
	    model->NBJTmodName, inst->NBJTname, s->real, s->imag);
	fprintf(stdout, "Ycc: % .5g,% .5g\n",
	    yIcVce.real, yIcVce.imag);
	fprintf(stdout, "Ycb: % .5g,% .5g\n",
	    yIcVbe.real, yIcVbe.imag);
	fprintf(stdout, "Ybc: % .5g,% .5g\n",
	    yIeVce.real - yIcVce.real, yIeVce.imag - yIcVce.imag);
	fprintf(stdout, "Ybb: % .5g,% .5g\n",
	    yIeVbe.real - yIcVbe.real, yIeVbe.imag - yIcVbe.imag);
      }
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

      inst->NBJTpDevice->pStats->totalTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
