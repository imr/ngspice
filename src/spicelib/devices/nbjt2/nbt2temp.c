/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "nbjt2def.h"
#include "numenum.h"
#include "carddefs.h"
#include "sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "cidersupt.h"
#include "suffix.h"

#define NIL(type)   ((type *)0)

int
NBJT2temp(inModel, ckt)
  GENmodel *inModel;
  register CKTcircuit *ckt;
/*
 * perform the temperature update
 */
{
  register NBJT2model *model = (NBJT2model *) inModel;
  register NBJT2instance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  TWOmaterial *pM, *pMaterial, *pNextMaterial;
  double startTime;


  /* loop through all the models */
  for (; model != NULL; model = model->NBJT2nextModel) {
    methods = model->NBJT2methods;
    models = model->NBJT2models;
    options = model->NBJT2options;
    outputs = model->NBJT2outputs;

    if (!options->OPTNtnomGiven) {
      options->OPTNtnom = ckt->CKTnomTemp;
    }
    for (pM = model->NBJT2matlInfo; pM != NIL(TWOmaterial);
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

    for (inst = model->NBJT2instances; inst != NULL;
	inst = inst->NBJT2nextInstance) {
      if (inst->NBJT2owner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NBJT2tempGiven) {
	inst->NBJT2temp = ckt->CKTtemp;
      }
      if (!inst->NBJT2areaGiven || inst->NBJT2area <= 0.0) {
	inst->NBJT2area = 1.0;
      }
      if (!inst->NBJT2widthGiven || inst->NBJT2width <= 0.0) {
	inst->NBJT2width = 1.0;
      }
      inst->NBJT2pDevice->width =
	  inst->NBJT2area * inst->NBJT2width * options->OPTNdefw;

      /* Compute and save globals for this instance. */
      GLOBcomputeGlobals(&(inst->NBJT2globals), inst->NBJT2temp);

      /* Calculate new sets of material parameters. */
      pM = model->NBJT2matlInfo;
      pMaterial = inst->NBJT2pDevice->pMaterials;
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
      TWOsetDoping(inst->NBJT2pDevice, model->NBJT2profiles,
	  model->NBJT2dopTables);

      /* Assign physical parameters to the mesh. */
      TWOsetup(inst->NBJT2pDevice);

      /* Assign boundary condition parameters. */
      TWOsetBCparams(inst->NBJT2pDevice, model->NBJT2boundaries);

      /* Normalize everything. */
      TWOnormalize(inst->NBJT2pDevice);

      /* Find the device's type. */
      if (inst->NBJT2pDevice->pFirstContact->pNodes[0]->netConc < 0.0) {
	inst->NBJT2type = PNP;
	if (OneCarrier) {
	  methods->METHoneCarrier = P_TYPE;
	}
      } else {
	inst->NBJT2type = NPN;
	if (OneCarrier) {
	  methods->METHoneCarrier = N_TYPE;
	}
      }

      inst->NBJT2pDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
