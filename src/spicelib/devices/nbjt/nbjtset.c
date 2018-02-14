/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "nbjtdefs.h"
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
NBJTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
/*
 * load the diode structure with those pointers needed later for fast matrix
 * loading
 */
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  METHcard *methods;
  MODLcard *models;
  OPTNcard *options;
  OUTPcard *outputs;
  int error;
  int xMeshSize;
  ONEdevice *pDevice;
  ONEcoord *xCoordList = NULL;
  ONEdomain *domainList = NULL;
  DOPprofile *profileList = NULL;
  DOPtable *dopTableList = NULL;
  ONEmaterial *pM, *pMaterial = NULL, *materialList = NULL;
  double startTime;


  /* loop through all the diode models */
  for (; model != NULL; model = NBJTnextModel(model)) {
    if (!model->NBJTpInfo) {
      TSCALLOC(model->NBJTpInfo, 1, ONEtranInfo);
    }
    methods = model->NBJTmethods;
    if (!methods) {
      TSCALLOC(methods, 1, METHcard);
      model->NBJTmethods = methods;
    }
    models = model->NBJTmodels;
    if (!models) {
      TSCALLOC(models, 1, MODLcard);
      model->NBJTmodels = models;
    }
    options = model->NBJToptions;
    if (!options) {
      TSCALLOC(options, 1, OPTNcard);
      model->NBJToptions = options;
    }
    outputs = model->NBJToutputs;
    if (!outputs) {
      TSCALLOC(outputs, 1, OUTPcard);
      model->NBJToutputs = outputs;
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
    if (!options->OPTNbaseLengthGiven) {
      options->OPTNbaseLength = 0.0;
    }
    if (!options->OPTNbaseAreaGiven) {
      options->OPTNbaseArea = 1.0;
    }
    if (!options->OPTNdeviceTypeGiven) {
      options->OPTNdeviceType = OPTN_BIPOLAR;
    }
    if (!options->OPTNicFileGiven) {
      options->OPTNicFile = NULL;
      options->OPTNunique = FALSE;		/* Can't form a unique name. */
    }
    if (!options->OPTNuniqueGiven) {
      options->OPTNunique = FALSE;
    }

    /* Set up the rest of the card lists */
    if ((error = MODLsetup(model->NBJTmodels)) != 0)
      return (error);
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    if ((error = OUTPsetup(model->NBJToutputs)) != 0)
      return (error);
    if ((error = MATLsetup(model->NBJTmaterials, &materialList)) != 0)
      return (error);
    if ((error = MOBsetup(model->NBJTmobility, materialList)) != 0)
      return (error);
    if ((error = MESHsetup('x', model->NBJTxMeshes, &xCoordList, &xMeshSize)) != 0)
      return (error);
    if ((error = DOMNsetup(model->NBJTdomains, &domainList,
	    xCoordList, NULL, materialList)) != 0)
      return (error);
    if ((error = BDRYsetup(model->NBJTboundaries,
	    xCoordList, NULL, domainList)) != 0)
      return (error);
    if ((error = CONTsetup(model->NBJTcontacts, NULL)) != 0)
      return (error);
    if ((error = DOPsetup(model->NBJTdopings, &profileList,
	    &dopTableList, xCoordList, NULL)) != 0)
      return (error);
    model->NBJTmatlInfo = materialList;
    model->NBJTprofiles = profileList;
    model->NBJTdopTables = dopTableList;

    /* loop through all the instances of the model */
    for (inst = NBJTinstances(model); inst != NULL;
         inst = NBJTnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NBJTprintGiven) {
	inst->NBJTprint = 0;
      } else if (inst->NBJTprint <= 0) {
	inst->NBJTprint = 1;
      }
      if (!inst->NBJTicFileGiven) {
	if (options->OPTNunique) {
	  inst->NBJTicFile = tprintf("%s.%s", options->OPTNicFile, inst->NBJTname);
	} else if (options->OPTNicFile != NULL) {
	  inst->NBJTicFile = tprintf("%s", options->OPTNicFile);
	} else {
	  inst->NBJTicFile = NULL;
	}
      }
      inst->NBJTstate = *states;
      *states += NBJTnumStates;

      if (!inst->NBJTpDevice) {
	/* Assign the mesh info to each instance. */
	TSCALLOC(pDevice, 1, ONEdevice);
	TSCALLOC(pDevice->pStats, 1, ONEstats);
	pDevice->name = inst->NBJTname;
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

	if (options->OPTNbaseDepthGiven) {
	  /* The base contact depth has been specified in the input. */
	  pDevice->baseIndex = MESHlocate(xCoordList, options->OPTNbaseDepth);
	} else {
	  pDevice->baseIndex = -1;	/* Invalid index acts as a flag */
	}
	/* store the device info in the instance */
	inst->NBJTpDevice = pDevice;
      }
      /* Now update the state pointers. */
      ONEgetStatePointers(inst->NBJTpDevice, states);

      /* Wipe out statistics from previous runs (if any). */
      memset(inst->NBJTpDevice->pStats, 0, sizeof(ONEstats));

      inst->NBJTpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if ((inst->ptr = SMPmakeElt(matrix, inst->first, inst->second)) == NULL){\
  return(E_NOMEM);\
} } while(0)

      TSTALLOC(NBJTcolColPtr, NBJTcolNode, NBJTcolNode);
      TSTALLOC(NBJTbaseBasePtr, NBJTbaseNode, NBJTbaseNode);
      TSTALLOC(NBJTemitEmitPtr, NBJTemitNode, NBJTemitNode);
      TSTALLOC(NBJTcolBasePtr, NBJTcolNode, NBJTbaseNode);
      TSTALLOC(NBJTcolEmitPtr, NBJTcolNode, NBJTemitNode);
      TSTALLOC(NBJTbaseColPtr, NBJTbaseNode, NBJTcolNode);
      TSTALLOC(NBJTbaseEmitPtr, NBJTbaseNode, NBJTemitNode);
      TSTALLOC(NBJTemitColPtr, NBJTemitNode, NBJTcolNode);
      TSTALLOC(NBJTemitBasePtr, NBJTemitNode, NBJTbaseNode);
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killDomainInfo(domainList);
  }
  return (OK);
}
