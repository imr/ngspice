/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "numosdef.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/meshext.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/ciderinp.h"
#include "ngspice/suffix.h"

#define TSCALLOC(var, size, type)\
if (size && (var = (type *)calloc(1, (unsigned)(size)*sizeof(type))) == NULL) {\
   return(E_NOMEM);\
}

int
NUMOSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
/*
 * load the structure with those pointers needed later for fast matrix
 * loading
 */
{
  register NUMOSmodel *model = (NUMOSmodel *) inModel;
  register NUMOSinstance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  int error, xIndex;
  int xMeshSize, yMeshSize;
  TWOdevice *pDevice;
  TWOcoord *xCoordList = NULL;
  TWOcoord *yCoordList = NULL;
  TWOdomain *domainList = NULL;
  TWOelectrode *electrodeList = NULL;
  TWOmaterial *pM, *pMaterial = NULL, *materialList = NULL;
  DOPprofile *profileList = NULL;
  DOPtable *dopTableList = NULL;
  double startTime;

  /* loop through all the models */
  for (; model != NULL; model = NUMOSnextModel(model)) {
    if (!model->NUMOSpInfo) {
      TSCALLOC(model->NUMOSpInfo, 1, TWOtranInfo);
    }
    methods = model->NUMOSmethods;
    if (!methods) {
      TSCALLOC(methods, 1, METHcard);
      model->NUMOSmethods = methods;
    }
    models = model->NUMOSmodels;
    if (!models) {
      TSCALLOC(models, 1, MODLcard);
      model->NUMOSmodels = models;
    }
    options = model->NUMOSoptions;
    if (!options) {
      TSCALLOC(options, 1, OPTNcard);
      model->NUMOSoptions = options;
    }
    outputs = model->NUMOSoutputs;
    if (!outputs) {
      TSCALLOC(outputs, 1, OUTPcard);
      model->NUMOSoutputs = outputs;
    }
    if (!methods->METHvoltPredGiven) {
      methods->METHvoltPred = FALSE;
    }
    if (!methods->METHmobDerivGiven) {
      methods->METHmobDeriv = TRUE;
    }
    if (!methods->METHoneCarrierGiven) {
      methods->METHoneCarrier = FALSE;
    }
    if (!methods->METHacAnalysisMethodGiven) {
      methods->METHacAnalysisMethod = SOR;
    }
    if (!methods->METHdabstolGiven) {
      methods->METHdabstol = DABSTOL2D;
    }
    if (!methods->METHdreltolGiven) {
      methods->METHdreltol = ckt->CKTreltol;
    }
    if (!methods->METHitLimGiven) {
      methods->METHitLim = 50;
    }
    if (!methods->METHomegaGiven || methods->METHomega <= 0.0) {
      methods->METHomega = 2.0 * M_PI /* radians/sec */ ;
    }
    if (!options->OPTNdefaGiven || options->OPTNdefa <= 0.0) {
      options->OPTNdefa = 1.0e4 /* cm^2 */ ;
    }
    if (!options->OPTNdeflGiven || options->OPTNdefl <= 0.0) {
      options->OPTNdefl = 1.0e2 /* cm */ ;
    }
    if (!options->OPTNdefwGiven && options->OPTNdefaGiven) {
      options->OPTNdefw = options->OPTNdefa / options->OPTNdefl;
    } else if (!options->OPTNdefwGiven || options->OPTNdefw <= 0.0) {
      options->OPTNdefw = 1.0e2 /* cm */ ;
    }
    if (!options->OPTNdeviceTypeGiven) {
      options->OPTNdeviceType = OPTN_MOSFET;
    }
    if (!options->OPTNicFileGiven) {
      options->OPTNicFile = NULL;
      options->OPTNunique = FALSE;		/* Can't form a unique name. */
    }
    if (!options->OPTNuniqueGiven) {
      options->OPTNunique = FALSE;
    }
    OneCarrier = methods->METHoneCarrier;

    /* Set up the rest of the card lists */
    if ((error = MODLsetup(model->NUMOSmodels)) != 0)
      return (error);
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;
    SurfaceMobility = models->MODLsurfaceMobility;

    if ((error = OUTPsetup(model->NUMOSoutputs)) != 0)
      return (error);
    if ((error = MATLsetup(model->NUMOSmaterials, &materialList)) != 0)
      return (error);
    if ((error = MOBsetup(model->NUMOSmobility, materialList)) != 0)
      return (error);
    if ((error = MESHsetup('x', model->NUMOSxMeshes, &xCoordList, &xMeshSize)) != 0)
      return (error);
    if ((error = MESHsetup('y', model->NUMOSyMeshes, &yCoordList, &yMeshSize)) != 0)
      return (error);
    if ((error = DOMNsetup(model->NUMOSdomains, &domainList,
	    xCoordList, yCoordList, materialList)) != 0)
      return (error);
    if ((error = BDRYsetup(model->NUMOSboundaries,
	    xCoordList, yCoordList, domainList)) != 0)
      return (error);
    if ((error = ELCTsetup(model->NUMOSelectrodes, &electrodeList,
	    xCoordList, yCoordList)) != 0)
      return (error);
    /* Make sure electrodes are OK. */
    checkElectrodes(electrodeList, 4);	/* NUMOS has 4 electrodes */

    if ((error = CONTsetup(model->NUMOScontacts, electrodeList)) != 0)
      return (error);
    if ((error = DOPsetup(model->NUMOSdopings, &profileList,
	    &dopTableList, xCoordList, yCoordList)) != 0)
      return (error);
    model->NUMOSmatlInfo = materialList;
    model->NUMOSprofiles = profileList;
    model->NUMOSdopTables = dopTableList;

    /* loop through all the instances of the model */
    for (inst = NUMOSinstances(model); inst != NULL;
         inst = NUMOSnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NUMOSprintGiven) {
	inst->NUMOSprint = 0;
      } else if (inst->NUMOSprint <= 0) {
	inst->NUMOSprint = 1;
      }
      if (!inst->NUMOSicFileGiven) {
	if (options->OPTNunique) {
	  inst->NUMOSicFile = tprintf("%s.%s", options->OPTNicFile, inst->NUMOSname);
	} else if (options->OPTNicFile != NULL) {
	  inst->NUMOSicFile = tprintf("%s", options->OPTNicFile);
	} else {
	  inst->NUMOSicFile = NULL;
	}
      }
      inst->NUMOSstate = *states;
      *states += NUMOSnumStates;

      if (!inst->NUMOSpDevice) {
	/* Assign the mesh info to each instance. */
	TSCALLOC(pDevice, 1, TWOdevice);
	TSCALLOC(pDevice->pStats, 1, TWOstats);
	pDevice->name = inst->NUMOSname;
	pDevice->solverType = SLV_NONE;
	pDevice->numXNodes = xMeshSize;
	pDevice->numYNodes = yMeshSize;
	pDevice->xScale = MESHmkArray(xCoordList, xMeshSize);
	pDevice->yScale = MESHmkArray(yCoordList, yMeshSize);
	pDevice->abstol = methods->METHdabstol;
	pDevice->reltol = methods->METHdreltol;
	TSCALLOC(pDevice->elemArray, pDevice->numXNodes, TWOelem **);
	for (xIndex = 1; xIndex < pDevice->numXNodes; xIndex++) {
	  TSCALLOC(pDevice->elemArray[xIndex], pDevice->numYNodes, TWOelem *);
	}

	/* Create a copy of material data that can change with temperature. */
	pDevice->pMaterials = NULL;
	for (pM = materialList; pM != NULL; pM = pM->next) {
	  if (pDevice->pMaterials == NULL) {
	    TSCALLOC(pMaterial, 1, TWOmaterial);
	    pDevice->pMaterials = pMaterial;
	  } else {
	    TSCALLOC(pMaterial->next, 1, TWOmaterial);
	    pMaterial = pMaterial->next;
	  }
	  /* Copy everything, then fix the incorrect pointer. */
	  memcpy(pMaterial, pM, sizeof(TWOmaterial));
	  pMaterial->next = NULL;
	}

	/* Generate the mesh structure for the device. */
	TWObuildMesh(pDevice, domainList, electrodeList, pDevice->pMaterials);

	/* Store the device info in the instance. */
	inst->NUMOSpDevice = pDevice;
      }
      /* Now update the state pointers. */
      TWOgetStatePointers(inst->NUMOSpDevice, states);

      /* Wipe out statistics from previous runs (if any). */
      memset(inst->NUMOSpDevice->pStats, 0, sizeof(TWOstats));

      inst->NUMOSpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if ((inst->ptr = SMPmakeElt(matrix, inst->first, inst->second)) == NULL){\
  return(E_NOMEM);\
} } while(0)

      TSTALLOC(NUMOSdrainDrainPtr, NUMOSdrainNode, NUMOSdrainNode);
      TSTALLOC(NUMOSdrainSourcePtr, NUMOSdrainNode, NUMOSsourceNode);
      TSTALLOC(NUMOSdrainGatePtr, NUMOSdrainNode, NUMOSgateNode);
      TSTALLOC(NUMOSdrainBulkPtr, NUMOSdrainNode, NUMOSbulkNode);
      TSTALLOC(NUMOSsourceDrainPtr, NUMOSsourceNode, NUMOSdrainNode);
      TSTALLOC(NUMOSsourceSourcePtr, NUMOSsourceNode, NUMOSsourceNode);
      TSTALLOC(NUMOSsourceGatePtr, NUMOSsourceNode, NUMOSgateNode);
      TSTALLOC(NUMOSsourceBulkPtr, NUMOSsourceNode, NUMOSbulkNode);
      TSTALLOC(NUMOSgateDrainPtr, NUMOSgateNode, NUMOSdrainNode);
      TSTALLOC(NUMOSgateSourcePtr, NUMOSgateNode, NUMOSsourceNode);
      TSTALLOC(NUMOSgateGatePtr, NUMOSgateNode, NUMOSgateNode);
      TSTALLOC(NUMOSgateBulkPtr, NUMOSgateNode, NUMOSbulkNode);
      TSTALLOC(NUMOSbulkDrainPtr, NUMOSbulkNode, NUMOSdrainNode);
      TSTALLOC(NUMOSbulkSourcePtr, NUMOSbulkNode, NUMOSsourceNode);
      TSTALLOC(NUMOSbulkGatePtr, NUMOSbulkNode, NUMOSgateNode);
      TSTALLOC(NUMOSbulkBulkPtr, NUMOSbulkNode, NUMOSbulkNode);
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killCoordInfo(yCoordList);
    killDomainInfo(domainList);
    killElectrodeInfo(electrodeList);
  }
  return (OK);
}
