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

#define NIL(type)   ((type *)0)
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
  char *icFileName = NULL;
  size_t nameLen;
  int error, xIndex;
  int xMeshSize, yMeshSize;
  TWOdevice *pDevice;
  TWOcoord *xCoordList = NIL(TWOcoord);
  TWOcoord *yCoordList = NIL(TWOcoord);
  TWOdomain *domainList = NIL(TWOdomain);
  TWOelectrode *electrodeList = NIL(TWOelectrode);
  TWOmaterial *pM, *pMaterial = NIL(TWOmaterial), *materialList = NIL(TWOmaterial);
  DOPprofile *profileList = NIL(DOPprofile);
  DOPtable *dopTableList = NIL(DOPtable);
  double startTime;

  /* loop through all the models */
  for (; model != NULL; model = model->NUMOSnextModel) {
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
    for (inst = model->NUMOSinstances; inst != NULL;
	inst = inst->NUMOSnextInstance) {

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NUMOSprintGiven) {
	inst->NUMOSprint = 0;
      } else if (inst->NUMOSprint <= 0) {
	inst->NUMOSprint = 1;
      }
      if (!inst->NUMOSicFileGiven) {
	if (options->OPTNunique) {
	  nameLen = strlen(options->OPTNicFile) + strlen(inst->NUMOSname) + 1;
	  TSCALLOC(icFileName, nameLen+1, char);
	  sprintf(icFileName, "%s.%s", options->OPTNicFile, inst->NUMOSname);
	  icFileName[nameLen] = '\0';
          inst->NUMOSicFile = icFileName;
	} else if (options->OPTNicFile != NULL) {
	  nameLen = strlen(options->OPTNicFile);
	  TSCALLOC(icFileName, nameLen+1, char);
	  icFileName = strcpy(icFileName, options->OPTNicFile);
	  inst->NUMOSicFile = icFileName;
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
	pDevice->pMaterials = NIL(TWOmaterial);
	for (pM = materialList; pM != NIL(TWOmaterial); pM = pM->next) {
	  if (pDevice->pMaterials == NIL(TWOmaterial)) {
	    TSCALLOC(pMaterial, 1, TWOmaterial);
	    pDevice->pMaterials = pMaterial;
	  } else {
	    TSCALLOC(pMaterial->next, 1, TWOmaterial);
	    pMaterial = pMaterial->next;
	  }
	  /* Copy everything, then fix the incorrect pointer. */
	  bcopy(pM, pMaterial, sizeof(TWOmaterial));
	  pMaterial->next = NIL(TWOmaterial);
	}

	/* Generate the mesh structure for the device. */
	TWObuildMesh(pDevice, domainList, electrodeList, pDevice->pMaterials);

	/* Store the device info in the instance. */
	inst->NUMOSpDevice = pDevice;
      }
      /* Now update the state pointers. */
      TWOgetStatePointers(inst->NUMOSpDevice, states);

      /* Wipe out statistics from previous runs (if any). */
      bzero(inst->NUMOSpDevice->pStats, sizeof(TWOstats));

      inst->NUMOSpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if ((inst->ptr = SMPmakeElt(matrix, inst->first, inst->second)) == NULL){\
  return(E_NOMEM);\
}

      TSTALLOC(NUMOSdrainDrainPtr, NUMOSdrainNode, NUMOSdrainNode)
      TSTALLOC(NUMOSdrainSourcePtr, NUMOSdrainNode, NUMOSsourceNode)
      TSTALLOC(NUMOSdrainGatePtr, NUMOSdrainNode, NUMOSgateNode)
      TSTALLOC(NUMOSdrainBulkPtr, NUMOSdrainNode, NUMOSbulkNode)
      TSTALLOC(NUMOSsourceDrainPtr, NUMOSsourceNode, NUMOSdrainNode)
      TSTALLOC(NUMOSsourceSourcePtr, NUMOSsourceNode, NUMOSsourceNode)
      TSTALLOC(NUMOSsourceGatePtr, NUMOSsourceNode, NUMOSgateNode)
      TSTALLOC(NUMOSsourceBulkPtr, NUMOSsourceNode, NUMOSbulkNode)
      TSTALLOC(NUMOSgateDrainPtr, NUMOSgateNode, NUMOSdrainNode)
      TSTALLOC(NUMOSgateSourcePtr, NUMOSgateNode, NUMOSsourceNode)
      TSTALLOC(NUMOSgateGatePtr, NUMOSgateNode, NUMOSgateNode)
      TSTALLOC(NUMOSgateBulkPtr, NUMOSgateNode, NUMOSbulkNode)
      TSTALLOC(NUMOSbulkDrainPtr, NUMOSbulkNode, NUMOSdrainNode)
      TSTALLOC(NUMOSbulkSourcePtr, NUMOSbulkNode, NUMOSsourceNode)
      TSTALLOC(NUMOSbulkGatePtr, NUMOSbulkNode, NUMOSgateNode)
      TSTALLOC(NUMOSbulkBulkPtr, NUMOSbulkNode, NUMOSbulkNode)
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killCoordInfo(yCoordList);
    killDomainInfo(domainList);
    killElectrodeInfo(electrodeList);
  }
  return (OK);
}
