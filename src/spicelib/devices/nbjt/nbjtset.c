/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "nbjtdefs.h"
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
NBJTsetup(matrix, inModel, ckt, states)
  register SMPmatrix *matrix;
  GENmodel *inModel;
  CKTcircuit *ckt;
  int *states;
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
  char *icFileName = NULL;
  int nameLen;
  int error;
  int xMeshSize;
  ONEdevice *pDevice;
  ONEcoord *xCoordList = NIL(ONEcoord);
  ONEdomain *domainList = NIL(ONEdomain);
  DOPprofile *profileList = NIL(DOPprofile);
  DOPtable *dopTableList = NIL(DOPtable);
  ONEmaterial *pM, *pMaterial = NULL, *materialList = NIL(ONEmaterial);
  double startTime;


  /* loop through all the diode models */
  for (; model != NULL; model = model->NBJTnextModel) {
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
    if ((error = MODLsetup(model->NBJTmodels)))
      return (error);
    BandGapNarrowing = models->MODLbandGapNarrowing;
    ConcDepLifetime = models->MODLconcDepLifetime;
    TempDepMobility = models->MODLtempDepMobility;
    ConcDepMobility = models->MODLconcDepMobility;

    if ((error = OUTPsetup(model->NBJToutputs)))
      return (error);
    if ((error = MATLsetup(model->NBJTmaterials, &materialList)))
      return (error);
    if ((error = MOBsetup(model->NBJTmobility, materialList)))
      return (error);
    if ((error = MESHsetup('x', model->NBJTxMeshes, &xCoordList, &xMeshSize)))
      return (error);
    if ((error = DOMNsetup(model->NBJTdomains, &domainList,
	    xCoordList, NIL(ONEcoord), materialList)))
      return (error);
    if ((error = BDRYsetup(model->NBJTboundaries,
	    xCoordList, NIL(ONEcoord), domainList)))
      return (error);
    if ((error = CONTsetup(model->NBJTcontacts, NULL)))
      return (error);
    if ((error = DOPsetup(model->NBJTdopings, &profileList,
	    &dopTableList, xCoordList, NIL(ONEcoord))))
      return (error);
    model->NBJTmatlInfo = materialList;
    model->NBJTprofiles = profileList;
    model->NBJTdopTables = dopTableList;

    /* loop through all the instances of the model */
    for (inst = model->NBJTinstances; inst != NULL;
	inst = inst->NBJTnextInstance) {
      if (inst->NBJTowner != ARCHme) goto matrixpointers;

      startTime = SPfrontEnd->IFseconds();

      if (!inst->NBJTprintGiven) {
	inst->NBJTprint = 0;
      } else if (inst->NBJTprint <= 0) {
	inst->NBJTprint = 1;
      }
      if (!inst->NBJTicFileGiven) {
	if (options->OPTNunique) {
	  nameLen = strlen(options->OPTNicFile) + strlen(inst->NBJTname) + 1;
	  TSCALLOC(icFileName, nameLen+1, char);
	  sprintf(icFileName, "%s.%s", options->OPTNicFile, inst->NBJTname);
	  icFileName[nameLen] = '\0';
          inst->NBJTicFile = icFileName;
	} else if (options->OPTNicFile != NULL) {
	  nameLen = strlen(options->OPTNicFile);
	  TSCALLOC(icFileName, nameLen+1, char);
	  icFileName = strcpy(icFileName, options->OPTNicFile);
	  inst->NBJTicFile = icFileName;
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
	  bcopy((char *) pM, (char *) pMaterial, sizeof(ONEmaterial));
	  pMaterial->next = NIL(ONEmaterial);
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
      bzero((char *) inst->NBJTpDevice->pStats, sizeof(ONEstats));

      inst->NBJTpDevice->pStats->totalTime[STAT_SETUP] +=
	  SPfrontEnd->IFseconds() - startTime;

      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if ((inst->ptr = SMPmakeElt(matrix,inst->first,inst->second))==(double *)NULL){\
  return(E_NOMEM);\
}

matrixpointers:
      TSTALLOC(NBJTcolColPtr, NBJTcolNode, NBJTcolNode)
      TSTALLOC(NBJTbaseBasePtr, NBJTbaseNode, NBJTbaseNode)
      TSTALLOC(NBJTemitEmitPtr, NBJTemitNode, NBJTemitNode)
      TSTALLOC(NBJTcolBasePtr, NBJTcolNode, NBJTbaseNode)
      TSTALLOC(NBJTcolEmitPtr, NBJTcolNode, NBJTemitNode)
      TSTALLOC(NBJTbaseColPtr, NBJTbaseNode, NBJTcolNode)
      TSTALLOC(NBJTbaseEmitPtr, NBJTbaseNode, NBJTemitNode)
      TSTALLOC(NBJTemitColPtr, NBJTemitNode, NBJTcolNode)
      TSTALLOC(NBJTemitBasePtr, NBJTemitNode, NBJTbaseNode)
    }
    /* Clean up lists */
    killCoordInfo(xCoordList);
    killDomainInfo(domainList);
  }
  return (OK);
}
