/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "numddefs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/ciderinp.h"
#include "ngspice/suffix.h"
#include "ngspice/meshext.h"

#define TSCALLOC(var, size, type)\
if (size && (var =(type *)calloc(1, (unsigned)(size)*sizeof(type))) == NULL) {\
   return(E_NOMEM);\
}


int 
NUMDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
/*
 * load the structure with those pointers needed later for fast matrix
 * loading
 */
{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  int error;
  int xMeshSize;
  ONEdevice *pDevice;
  ONEcoord *xCoordList = NULL;
  ONEdomain *domainList = NULL;
  ONEmaterial *pM, *pMaterial = NULL, *materialList = NULL;
  DOPprofile *profileList = NULL;
  DOPtable *dopTableList = NULL;
  double startTime;


  /* loop through all the models */
  for (; model != NULL; model = NUMDnextModel(model)) {
    if (!model->NUMDpInfo) {
      TSCALLOC(model->NUMDpInfo, 1, ONEtranInfo);
    }
    methods = model->NUMDmethods;
    if (!methods) {
      TSCALLOC(methods, 1, METHcard);
      model->NUMDmethods = methods;
    }
    models = model->NUMDmodels;
    if (!models) {
      TSCALLOC(models, 1, MODLcard);
      model->NUMDmodels = models;
    }
    options = model->NUMDoptions;
    if (!options) {
      TSCALLOC(options, 1, OPTNcard);
      model->NUMDoptions = options;
    }
    outputs = model->NUMDoutputs;
    if (!outputs) {
      TSCALLOC(outputs, 1, OUTPcard);
      model->NUMDoutputs = outputs;
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
      methods->METHdabstol = DABSTOL1D;
    }
    if (!methods->METHdreltolGiven) {
      methods->METHdreltol = ckt->CKTreltol;
    }
    if (!methods->METHitLimGiven) {
      methods->METHitLim = 20;
    }
    if (!methods->METHomegaGiven || methods->METHomega <= 0.0) {
      methods->METHomega = 2.0 * M_PI /* radians/sec */ ;
    }
    if (!options->OPTNdefaGiven || options->OPTNdefa <= 0.0) {
      options->OPTNdefa = 1.0e4 /* cm^2 */ ;
    }
    if (!options->OPTNdeviceTypeGiven) {
      options->OPTNdeviceType = OPTN_DIODE;
    }
    if (!options->OPTNicFileGiven) {
      options->OPTNicFile = NULL;
      options->OPTNunique = FALSE;		/* Can't form a unique name. */
    }
    if (!options->OPTNuniqueGiven) {
      options->OPTNunique = FALSE;
    }

    /* Set up the rest of the card lists */
    if ((error = MODLsetup(model->NUMDmodels)) != 0)
      return (error);
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    if ((error = OUTPsetup(model->NUMDoutputs)) != 0)
      return (error);
    if ((error = MATLsetup(model->NUMDmaterials, &materialList)) != 0)
      return (error);
    if ((error = MOBsetup(model->NUMDmobility, materialList)) != 0)
      return (error);
    if ((error = MESHsetup('x', model->NUMDxMeshes, &xCoordList, &xMeshSize)) != 0)
      return (error);
    if ((error = DOMNsetup(model->NUMDdomains, &domainList,
	    xCoordList, NULL, materialList)) != 0)
      return (error);
    if ((error = BDRYsetup(model->NUMDboundaries,
	    xCoordList, NULL, domainList)) != 0)
      return (error);
    if ((error = CONTsetup(model->NUMDcontacts, NULL)) != 0)
      return (error);
    if ((error = DOPsetup(model->NUMDdopings, &profileList,
	    &dopTableList, xCoordList, NULL)) != 0)
      return (error);
    model->NUMDmatlInfo = materialList;
    model->NUMDprofiles = profileList;
    model->NUMDdopTables = dopTableList;

    /* loop through all the instances of the model */
    for (inst = NUMDinstances(model); inst != NULL;
         inst = NUMDnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();

      if ((!inst->NUMDprintGiven)) {
	inst->NUMDprint = 0;
      } else if (inst->NUMDprint <= 0) {
	inst->NUMDprint = 1;
      }
      if ((!inst->NUMDicFileGiven)) {
	if (options->OPTNunique) {
	  inst->NUMDicFile = tprintf("%s.%s", options->OPTNicFile, inst->NUMDname);
	} else if (options->OPTNicFile != NULL) {
	  inst->NUMDicFile = tprintf("%s", options->OPTNicFile);
	} else {
	  inst->NUMDicFile = NULL;
	}
      }
      inst->NUMDstate = *states;
      *states += NUMDnumStates;

      if (!inst->NUMDpDevice) {
	/* Assign the mesh info to each instance. */
	TSCALLOC(pDevice, 1, ONEdevice);
	TSCALLOC(pDevice->pStats, 1, ONEstats);
	pDevice->name = inst->NUMDname;
	pDevice->solverType = SLV_NONE;
	pDevice->numNodes = xMeshSize;
	pDevice->abstol = methods->METHdabstol;
	pDevice->reltol = methods->METHdreltol;
	pDevice->rhsImag = NULL;
	TSCALLOC(pDevice->elemArray, pDevice->numNodes, ONEelem *);

	/* Create a copy of material data that can change with temperature. */
	pDevice->pMaterials = NULL;
	for (pM = materialList; pM != NULL; pM = pM->next) {
	  if (pDevice->pMaterials == NULL) {
	    TSCALLOC(pMaterial, 1, ONEmaterial);
	    pDevice->pMaterials = pMaterial;
	  } else {
	    TSCALLOC(pMaterial->next, 1, ONEmaterial);
	    pMaterial = pMaterial->next;
	  }
	  /* Copy everything, then fix the incorrect pointer. */
	  memcpy(pMaterial, pM, sizeof(ONEmaterial));
	  pMaterial->next = NULL;
	}

	/* generate the mesh structure for the device */
	ONEbuildMesh(pDevice, xCoordList, domainList, pDevice->pMaterials);

	/* store the device info in the instance */
	inst->NUMDpDevice = pDevice;
      }
      /* Now update the state pointers. */
      ONEgetStatePointers(inst->NUMDpDevice, states);

      /* Wipe out statistics from previous runs (if any). */
      memset(inst->NUMDpDevice->pStats, 0, sizeof(ONEstats));

      inst->NUMDpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if ((inst->ptr = SMPmakeElt(matrix, inst->first, inst->second)) == NULL){\
  return(E_NOMEM);\
} } while(0)

      TSTALLOC(NUMDposPosPtr, NUMDposNode, NUMDposNode);
      TSTALLOC(NUMDnegNegPtr, NUMDnegNode, NUMDnegNode);
      TSTALLOC(NUMDnegPosPtr, NUMDnegNode, NUMDposNode);
      TSTALLOC(NUMDposNegPtr, NUMDposNode, NUMDnegNode);
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killDomainInfo(domainList);
  }
  return (OK);
}
