/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "numddefs.h"
#include "numconst.h"
#include "numenum.h"
#include "sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "ciderinp.h"
#include "suffix.h"

#define NIL(type)   ((type *)0)
#define TSCALLOC(var, size, type)\
if (size && (!(var =(type *)calloc(1, (unsigned)(size)*sizeof(type))))) {\
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
  char *icFileName = NULL;
  int nameLen;
  int error;
  int xMeshSize;
  ONEdevice *pDevice;
  ONEcoord *xCoordList = NIL(ONEcoord);
  ONEdomain *domainList = NIL(ONEdomain);
  ONEmaterial *pM, *pMaterial = NIL(ONEmaterial), *materialList = NIL(ONEmaterial);
  DOPprofile *profileList = NIL(DOPprofile);
  DOPtable *dopTableList = NIL(DOPtable);
  double startTime;


  /* loop through all the models */
  for (; model != NULL; model = model->NUMDnextModel) {
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
    if ((error = MODLsetup(model->NUMDmodels)))
      return (error);
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    if ((error = OUTPsetup(model->NUMDoutputs)))
      return (error);
    if ((error = MATLsetup(model->NUMDmaterials, &materialList)))
      return (error);
    if ((error = MOBsetup(model->NUMDmobility, materialList)))
      return (error);
    if ((error = MESHsetup('x', model->NUMDxMeshes, &xCoordList, &xMeshSize)))
      return (error);
    if ((error = DOMNsetup(model->NUMDdomains, &domainList,
	    xCoordList, NIL(ONEcoord), materialList)))
      return (error);
    if ((error = BDRYsetup(model->NUMDboundaries,
	    xCoordList, NIL(ONEcoord), domainList)))
      return (error);
    if ((error = CONTsetup(model->NUMDcontacts, NULL)))
      return (error);
    if ((error = DOPsetup(model->NUMDdopings, &profileList,
	    &dopTableList, xCoordList, NIL(ONEcoord))))
      return (error);
    model->NUMDmatlInfo = materialList;
    model->NUMDprofiles = profileList;
    model->NUMDdopTables = dopTableList;

    /* loop through all the instances of the model */
    for (inst = model->NUMDinstances; inst != NULL;
	inst = inst->NUMDnextInstance) {
      if ((inst->NUMDowner != ARCHme)) goto matrixpointers;

      startTime = SPfrontEnd->IFseconds();

      if ((!inst->NUMDprintGiven)) {
	inst->NUMDprint = 0;
      } else if (inst->NUMDprint <= 0) {
	inst->NUMDprint = 1;
      }
      if ((!inst->NUMDicFileGiven)) {
	if (options->OPTNunique) {
	  nameLen = strlen(options->OPTNicFile) + strlen(inst->NUMDname) + 1;
	  TSCALLOC(icFileName, nameLen+1, char);
	  sprintf(icFileName, "%s.%s", options->OPTNicFile, inst->NUMDname);
	  icFileName[nameLen] = '\0';
          inst->NUMDicFile = icFileName;
	} else if (options->OPTNicFile != NULL) {
	  nameLen = strlen(options->OPTNicFile);
	  TSCALLOC(icFileName, nameLen+1, char);
	  icFileName = strcpy(icFileName, options->OPTNicFile);
	  inst->NUMDicFile = icFileName;
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
	pDevice->rhsImag = NIL(double);
	TSCALLOC(pDevice->elemArray, pDevice->numNodes, ONEelem *);

	/* Create a copy of material data that can change with temperature. */
	pDevice->pMaterials = NIL(ONEmaterial);
	for (pM = materialList; pM != NIL(ONEmaterial); pM = pM->next) {
	  if (pDevice->pMaterials == NIL(ONEmaterial)) {
	    TSCALLOC(pMaterial, 1, ONEmaterial);
	    pDevice->pMaterials = pMaterial;
	  } else {
	    TSCALLOC(pMaterial->next, 1, ONEmaterial);
	    pMaterial = pMaterial->next;
	  }
	  /* Copy everything, then fix the incorrect pointer. */
	  bcopy((void *) pM, (void *) pMaterial, sizeof(ONEmaterial));
	  pMaterial->next = NIL(ONEmaterial);
	}

	/* generate the mesh structure for the device */
	ONEbuildMesh(pDevice, xCoordList, domainList, pDevice->pMaterials);

	/* store the device info in the instance */
	inst->NUMDpDevice = pDevice;
      }
      /* Now update the state pointers. */
      ONEgetStatePointers(inst->NUMDpDevice, states);

      /* Wipe out statistics from previous runs (if any). */
      bzero((void *) inst->NUMDpDevice->pStats, sizeof(ONEstats));

      inst->NUMDpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if ((inst->ptr = SMPmakeElt(matrix,inst->first,inst->second))==(double *)NULL){\
  return(E_NOMEM);\
}

matrixpointers:
      TSTALLOC(NUMDposPosPtr, NUMDposNode, NUMDposNode)
      TSTALLOC(NUMDnegNegPtr, NUMDnegNode, NUMDnegNode)
      TSTALLOC(NUMDnegPosPtr, NUMDnegNode, NUMDposNode)
      TSTALLOC(NUMDposNegPtr, NUMDposNode, NUMDnegNode)
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killDomainInfo(domainList);
  }
  return (OK);
}
