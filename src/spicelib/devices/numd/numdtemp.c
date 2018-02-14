/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numddefs.h"
#include "ngspice/numenum.h"
#include "ngspice/carddefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int 
NUMDtemp(GENmodel *inModel, CKTcircuit *ckt)
/*
 * perform the temperature update to the diode
 */
{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  ONEmaterial *pM, *pMaterial, *pNextMaterial;
  double startTime;


  /* loop through all the diode models */
  for (; model != NULL; model = NUMDnextModel(model)) {
    methods = model->NUMDmethods;
    models = model->NUMDmodels;
    options = model->NUMDoptions;
    outputs = model->NUMDoutputs;

    if (!options->OPTNtnomGiven) {
      options->OPTNtnom = ckt->CKTnomTemp;
    }
    for (pM = model->NUMDmatlInfo; pM != NULL; pM = pM->next) {
      pM->tnom = options->OPTNtnom;
    }

    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    for (inst = NUMDinstances(model); inst != NULL;
         inst = NUMDnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NUMDtempGiven) {
	inst->NUMDtemp = ckt->CKTtemp;
      }
      if (!inst->NUMDareaGiven || inst->NUMDarea <= 0.0) {
	inst->NUMDarea = 1.0;
      }
      inst->NUMDpDevice->area = inst->NUMDarea * options->OPTNdefa;

      /* Compute and save globals for this instance. */
      GLOBcomputeGlobals(&(inst->NUMDglobals), inst->NUMDtemp);
      if (outputs->OUTPglobals) {
	GLOBprnGlobals(stdout, &(inst->NUMDglobals));
      }
      /* Calculate new sets of material parameters. */
      pM = model->NUMDmatlInfo;
      pMaterial = inst->NUMDpDevice->pMaterials;
      for (; pM != NULL; pM = pM->next, pMaterial = pMaterial->next) {

	/* Copy the original values, then fix the incorrect pointer. */
	pNextMaterial = pMaterial->next;
	memcpy(pMaterial, pM, sizeof(ONEmaterial));
	pMaterial->next = pNextMaterial;

	/* Now do the temperature dependence. */
	MATLtempDep(pMaterial, pMaterial->tnom);
	if (outputs->OUTPmaterial) {
	  printMaterialInfo(pMaterial);
	}
      }

      /* Assign doping to the mesh. */
      ONEsetDoping(inst->NUMDpDevice, model->NUMDprofiles,
	  model->NUMDdopTables);

      /* Assign other physical parameters to the mesh. */
      ONEsetup(inst->NUMDpDevice);

      /* Assign boundary condition parameters. */
      ONEsetBCparams(inst->NUMDpDevice, model->NUMDboundaries,
	  model->NUMDcontacts);

      /* Normalize everything. */
      ONEnormalize(inst->NUMDpDevice);

      /* Find the device's polarity type. */
      switch (options->OPTNdeviceType) {
      case OPTN_DIODE:
	if (inst->NUMDpDevice->elemArray[1]
	    ->pNodes[0]->netConc < 0.0) {
	  inst->NUMDtype = PN;
	} else {
	  inst->NUMDtype = NP;
	}
	break;
      case OPTN_MOSCAP:
	if (inst->NUMDpDevice->elemArray[inst->NUMDpDevice->numNodes - 1]
	    ->pNodes[1]->netConc < 0.0) {
	  inst->NUMDtype = PN;
	} else {
	  inst->NUMDtype = NP;
	}
	break;
      default:
	inst->NUMDtype = PN;
	break;
      }
      inst->NUMDpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
