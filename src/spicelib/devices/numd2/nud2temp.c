/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "numd2def.h"
#include "numenum.h"
#include "carddefs.h"
#include "sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "cidersupt.h"
#include "suffix.h"

#define NIL(type)   ((type *)0)

int
NUMD2temp(inModel, ckt)
  GENmodel *inModel;
  register CKTcircuit *ckt;
/*
 * perform the temperature update
 */
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  TWOmaterial *pM, *pMaterial, *pNextMaterial;
  double startTime;

  /* loop through all the models */
  for (; model != NULL; model = model->NUMD2nextModel) {
    methods = model->NUMD2methods;
    models = model->NUMD2models;
    options = model->NUMD2options;
    outputs = model->NUMD2outputs;

    if (!options->OPTNtnomGiven) {
      options->OPTNtnom = ckt->CKTnomTemp;
    }
    for (pM = model->NUMD2matlInfo; pM != NIL(TWOmaterial);
	pM = pM->next) {
      pM->tnom = options->OPTNtnom;
    }
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;
    SurfaceMobility = models->MODLsurfaceMobility;
    MatchingMobility = models->MODLmatchingMobility;
    OneCarrier = methods->METHoneCarrier;

    for (inst = model->NUMD2instances; inst != NULL;
	inst = inst->NUMD2nextInstance) {
      if (inst->NUMD2owner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NUMD2tempGiven) {
	inst->NUMD2temp = ckt->CKTtemp;
      }
      if (!inst->NUMD2areaGiven || inst->NUMD2area <= 0.0) {
	inst->NUMD2area = 1.0;
      }
      if (!inst->NUMD2widthGiven || inst->NUMD2width <= 0.0) {
	inst->NUMD2width = 1.0;
      }
      inst->NUMD2pDevice->width =
	  inst->NUMD2area * inst->NUMD2width * options->OPTNdefw;

      /* Compute and save globals for this instance. */
      GLOBcomputeGlobals(&(inst->NUMD2globals), inst->NUMD2temp);

      /* Calculate new sets of material parameters. */
      pM = model->NUMD2matlInfo;
      pMaterial = inst->NUMD2pDevice->pMaterials;
      for (; pM != NULL; pM = pM->next, pMaterial = pMaterial->next) {

	/* Copy everything, then fix the incorrect pointer. */
	pNextMaterial = pMaterial->next;
	bcopy((char *) pM, (char *) pMaterial, sizeof(TWOmaterial));
	pMaterial->next = pNextMaterial;

	/* Now do the temperature dependence. */
	MATLtempDep(pMaterial, pMaterial->tnom);
	if (outputs->OUTPmaterial) {
	  printMaterialInfo(pMaterial);
	}
      }

      /* Assign doping to the mesh. */
      TWOsetDoping(inst->NUMD2pDevice, model->NUMD2profiles,
	  model->NUMD2dopTables);

      /* Assign physical parameters to the mesh. */
      TWOsetup(inst->NUMD2pDevice);

      /* Assign boundary condition parameters. */
      TWOsetBCparams(inst->NUMD2pDevice, model->NUMD2boundaries);

      /* Normalize everything. */
      TWOnormalize(inst->NUMD2pDevice);

      /* Find the device's type. */
      if (inst->NUMD2pDevice->pFirstContact->pNodes[0]->netConc < 0.0) {
	inst->NUMD2type = PN;
	if (OneCarrier) {
	  methods->METHoneCarrier = P_TYPE;
	}
      } else {
	inst->NUMD2type = NP;
	if (OneCarrier) {
	  methods->METHoneCarrier = N_TYPE;
	}
      }
      inst->NUMD2pDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
