/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "nbjtdefs.h"
#include "numenum.h"
#include "carddefs.h"
#include "sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "suffix.h"

#define NIL(type)   ((type *)0)
extern int ONEdcDebug;
extern double LNorm;

int 
NBJTtemp(inModel, ckt)
  GENmodel *inModel;
  register CKTcircuit *ckt;
/*
 * perform the temperature update to the bjt
 */
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  ONEmaterial *pM, *pMaterial, *pNextMaterial;
  ONEdevice *pDevice;
  double startTime;
  int baseIndex, indexBE, indexBC;


  /* loop through all the bjt models */
  for (; model != NULL; model = model->NBJTnextModel) {
    methods = model->NBJTmethods;
    models = model->NBJTmodels;
    options = model->NBJToptions;
    outputs = model->NBJToutputs;

    if (!options->OPTNtnomGiven) {
      options->OPTNtnom = ckt->CKTnomTemp;
    }
    for (pM = model->NBJTmatlInfo; pM != NIL(ONEmaterial); pM = pM->next) {
      pM->tnom = options->OPTNtnom;
    }

    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    for (inst = model->NBJTinstances; inst != NULL;
	inst = inst->NBJTnextInstance) {
      if (inst->NBJTowner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NBJTtempGiven) {
	inst->NBJTtemp = ckt->CKTtemp;
      }
      if (!inst->NBJTareaGiven || inst->NBJTarea <= 0.0) {
	inst->NBJTarea = 1.0;
      }
      inst->NBJTpDevice->area = inst->NBJTarea * options->OPTNdefa;

      /* Compute and save globals for this instance. */
      GLOBcomputeGlobals(&(inst->NBJTglobals), inst->NBJTtemp);

      /* Calculate new sets of material parameters. */
      pM = model->NBJTmatlInfo;
      pMaterial = inst->NBJTpDevice->pMaterials;
      for (; pM != NULL; pM = pM->next, pMaterial = pMaterial->next) {

	/* Copy the original values, then fix the incorrect pointer. */
	pNextMaterial = pMaterial->next;
	bcopy((char *) pM, (char *) pMaterial, sizeof(ONEmaterial));
	pMaterial->next = pNextMaterial;

	/* Now do the temperature dependence. */
	MATLtempDep(pMaterial, pMaterial->tnom);
	if (outputs->OUTPmaterial) {
	  printMaterialInfo(pMaterial);
	}
      }

      /* Assign doping to the mesh. */
      ONEsetDoping(inst->NBJTpDevice, model->NBJTprofiles,
	  model->NBJTdopTables);

      /* Assign other physical parameters to the mesh. */
      ONEsetup(inst->NBJTpDevice);

      /* Assign boundary condition parameters. */
      ONEsetBCparams(inst->NBJTpDevice, model->NBJTboundaries,
	  model->NBJTcontacts);

      /* Normalize everything. */
      ONEnormalize(inst->NBJTpDevice);

      /* Find the device's type. */
      if (inst->NBJTpDevice->elemArray[1]->pNodes[0]->netConc < 0.0) {
	inst->NBJTtype = PNP;
      } else {
	inst->NBJTtype = NPN;
      }

      /* Find the location of the base index. */
      pDevice = inst->NBJTpDevice;
      baseIndex = pDevice->baseIndex;
      if (baseIndex <= 0) {
	if (options->OPTNbaseDepthGiven) {
	  printf("Warning: base contact not on node -- adjusting contact\n");
	}
	NBJTjunctions(pDevice, &indexBE, &indexBC);
	pDevice->baseIndex = 0.5 * (indexBE + indexBC);
      }
      if (inst->NBJTtype == PNP) {
	pDevice->elemArray[pDevice->baseIndex]->pNodes[0]->baseType = N_TYPE;
      } else if (inst->NBJTtype == NPN) {
	pDevice->elemArray[pDevice->baseIndex]->pNodes[0]->baseType = P_TYPE;
      } else {
	printf("NBJTtemp: unknown BJT type \n");
      }
      if (baseIndex <= 0 && !options->OPTNbaseDepthGiven) {
	ONEdcDebug = FALSE;
	ONEequilSolve(pDevice);
	adjustBaseContact(pDevice, indexBE, indexBC);
      }
      printf("BJT: base contact depth is %g um at node %d\n",
	  pDevice->elemArray[pDevice->baseIndex]->pNodes[0]->x * 1e4,
	  pDevice->baseIndex);

      /* Find, normalize and convert to reciprocal-form the base length. */
      pDevice->baseLength = options->OPTNbaseLength;
      if (pDevice->baseLength > 0.0) {
	pDevice->baseLength /= LNorm;
	pDevice->baseLength = 1.0 / pDevice->baseLength;
      } else if (pDevice->elemArray[pDevice->baseIndex]->evalNodes[0]) {
	pDevice->baseLength = pDevice->elemArray[pDevice->baseIndex]->rDx;
      } else {
	pDevice->baseLength = pDevice->elemArray[pDevice->baseIndex - 1]->rDx;
      }
      /* Adjust reciprocal base length to account for base area factor */
      pDevice->baseLength *= options->OPTNbaseArea;

      inst->NBJTpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

    }
  }
  return (OK);
}
