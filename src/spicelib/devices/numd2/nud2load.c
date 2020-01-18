/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This is the function called each iteration to evaluate the 2d numerical
 * Diodes in the circuit and load them into the matrix as appropriate
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "numd2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


/* Forward declarations */
int NUMD2initSmSig(NUMD2instance *);


int
NUMD2load(GENmodel *inModel, CKTcircuit *ckt)
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  register TWOdevice *pDevice;
  double startTime, startTime2, totalTime, totalTime2;
  double id=0.;
  double idhat = 0.0;
  double ideq;
  double gd=0.;
  double xfact;
  double tol;			/* temporary for tolerance calculations */
  double vd;			/* current diode voltage */
  double delVd;
  int Check;
  double deltaNorm[7];
  int devConverged = FALSE;
  int i;
  int numDevNonCon;
  int deviceType;
  int doInitSolve;
  int doVoltPred;
  char *initStateName;

  /* loop through all the diode models */
  for (; model != NULL; model = NUMD2nextModel(model)) {
    FieldDepMobility = model->NUMD2models->MODLfieldDepMobility;
    TransDepMobility = model->NUMD2models->MODLtransDepMobility;
    SurfaceMobility = model->NUMD2models->MODLsurfaceMobility;
    Srh = model->NUMD2models->MODLsrh;
    Auger = model->NUMD2models->MODLauger;
    AvalancheGen = model->NUMD2models->MODLavalancheGen;
    OneCarrier = model->NUMD2methods->METHoneCarrier;
    MobDeriv = model->NUMD2methods->METHmobDeriv;
    MaxIterations = model->NUMD2methods->METHitLim;
    TWOdcDebug = model->NUMD2outputs->OUTPdcDebug;
    TWOtranDebug = model->NUMD2outputs->OUTPtranDebug;
    TWOacDebug = model->NUMD2outputs->OUTPacDebug;
    deviceType = model->NUMD2options->OPTNdeviceType;
    doVoltPred = model->NUMD2methods->METHvoltPred;

    if (ckt->CKTmode & MODEINITPRED) {
      /* compute normalized deltas and predictor coeff */
      if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	model->NUMD2pInfo->order = ckt->CKTorder;
	model->NUMD2pInfo->method = ckt->CKTintegrateMethod;
	for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	  deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
	}
	computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMD2pInfo->intCoeff, deltaNorm);
	computePredCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMD2pInfo->predCoeff, deltaNorm);
      }
    } else if (ckt->CKTmode & MODEINITTRAN) {
      model->NUMD2pInfo->order = ckt->CKTorder;
/*      model->NUMD2pInfo->method = GEAR; */
      model->NUMD2pInfo->method = ckt->CKTintegrateMethod;
      for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
      }
      computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	  model->NUMD2pInfo->intCoeff, deltaNorm);
    }
    /* loop through all the instances of the model */
    for (inst = NUMD2instances(model); inst != NULL;
         inst = NUMD2nextInstance(inst)) {

      pDevice = inst->NUMD2pDevice;

      totalTime = 0.0;
      startTime = SPfrontEnd->IFseconds();

      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMD2globals));

      pDevice->devStates = ckt->CKTstates;
      /*
       * initialization
       */
      Check = 1;
      doInitSolve = FALSE;
      initStateName = NULL;
      if (ckt->CKTmode & MODEINITSMSIG) {
	vd = *(ckt->CKTstate0 + inst->NUMD2voltage);
	delVd = 0.0;
	NUMD2setBCs(pDevice, vd);
      } else if (ckt->CKTmode & MODEINITTRAN) {
	*(ckt->CKTstate0 + inst->NUMD2voltage) =
	    *(ckt->CKTstate1 + inst->NUMD2voltage);
	vd = *(ckt->CKTstate1 + inst->NUMD2voltage);
	TWOsaveState(pDevice);
	delVd = 0.0;
      } else if ((ckt->CKTmode & MODEINITJCT) &&
	  (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
	doInitSolve = TRUE;
	initStateName = inst->NUMD2icFile;
	vd = 0.0;
	delVd = vd;
      } else if ((ckt->CKTmode & MODEINITJCT) && inst->NUMD2off) {
	doInitSolve = TRUE;
	vd = 0.0;
	delVd = vd;
      } else if (ckt->CKTmode & MODEINITJCT) {
	doInitSolve = TRUE;
	initStateName = inst->NUMD2icFile;
	if (deviceType == OPTN_DIODE) {
	  vd = inst->NUMD2type * 0.6;
	} else if (deviceType == OPTN_MOSCAP) {
	  vd = inst->NUMD2type * 0.8;
	} else {
	  vd = 0.0;
	}
	delVd = vd;
      } else if (ckt->CKTmode & MODEINITFIX && inst->NUMD2off) {
	vd = 0.0;
	delVd = vd;
      } else {
	if (ckt->CKTmode & MODEINITPRED) {
	  *(ckt->CKTstate0 + inst->NUMD2voltage) =
	      *(ckt->CKTstate1 + inst->NUMD2voltage);
	  *(ckt->CKTstate0 + inst->NUMD2id) =
	      *(ckt->CKTstate1 + inst->NUMD2id);
	  *(ckt->CKTstate0 + inst->NUMD2conduct) =
	      *(ckt->CKTstate1 + inst->NUMD2conduct);
	  if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	    /* no linear prediction on device voltages */
	    vd = *(ckt->CKTstate1 + inst->NUMD2voltage);
	    TWOpredict(pDevice, model->NUMD2pInfo);
	  } else {
            if (doVoltPred) {
	      /* linear prediction */
	      xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
	      vd = (1+xfact) * (*(ckt->CKTstate1 + inst->NUMD2voltage))
		  -  (xfact) * (*(ckt->CKTstate2 + inst->NUMD2voltage));
	    } else {
	      vd = *(ckt->CKTstate1 + inst->NUMD2voltage);
	    }
	  }
	} else {
	  vd = *(ckt->CKTrhsOld + inst->NUMD2posNode) -
	      *(ckt->CKTrhsOld + inst->NUMD2negNode);
	}
	delVd = vd - *(ckt->CKTstate0 + inst->NUMD2voltage);
	idhat = *(ckt->CKTstate0 + inst->NUMD2id) +
	    *(ckt->CKTstate0 + inst->NUMD2conduct) * delVd;
	/*
	 * bypass if solution has not changed
	 */
	if ((ckt->CKTbypass) && pDevice->converged &&
	    !(ckt->CKTmode & MODEINITPRED)) {
	  tol = ckt->CKTvoltTol + ckt->CKTreltol *
	      MAX(fabs(vd), fabs(*(ckt->CKTstate0 + inst->NUMD2voltage)));
	  if (fabs(delVd) < tol) {
	    tol = ckt->CKTreltol *
		MAX(fabs(idhat), fabs(*(ckt->CKTstate0 + inst->NUMD2id))) +
		ckt->CKTabstol;
	    if (fabs(idhat - *(ckt->CKTstate0 + inst->NUMD2id))
		< tol) {
	      vd = *(ckt->CKTstate0 + inst->NUMD2voltage);
	      id = *(ckt->CKTstate0 + inst->NUMD2id);
	      gd = *(ckt->CKTstate0 + inst->NUMD2conduct);
	      goto load;
	    }
	  }
	}
	/*
	 * limit new junction voltage
	 */
	if (deviceType == OPTN_DIODE) {
	  vd = inst->NUMD2type * limitJunctionVoltage(
	      inst->NUMD2type * vd,
	      inst->NUMD2type * *(ckt->CKTstate0 + inst->NUMD2voltage),
	      &Check);
	} else if (deviceType == OPTN_MOSCAP) {
	  vd = inst->NUMD2type * limitVgb(
	      inst->NUMD2type * vd,
	      inst->NUMD2type * *(ckt->CKTstate0 + inst->NUMD2voltage),
	      &Check);
	} else {
	  vd = inst->NUMD2type * limitResistorVoltage(
	      inst->NUMD2type * vd,
	      inst->NUMD2type * *(ckt->CKTstate0 + inst->NUMD2voltage),
	      &Check);
	}
	delVd = vd - *(ckt->CKTstate0 + inst->NUMD2voltage);
	NUMD2setBCs(pDevice, vd - delVd);
      }
      if (doInitSolve) {
	if (TWOdcDebug) {
	  printVoltages(stdout,
	      model->NUMD2modName, inst->NUMD2name,
	      deviceType, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
	startTime2 = SPfrontEnd->IFseconds();
	TWOequilSolve(pDevice);
	totalTime2 = SPfrontEnd->IFseconds() - startTime2;
	pDevice->pStats->totalTime[STAT_SETUP] += totalTime2;
	pDevice->pStats->totalTime[STAT_DC] -= totalTime2;

	TWObiasSolve(pDevice, MaxIterations, FALSE, NULL);

	*(ckt->CKTstate0 + inst->NUMD2voltage) = 0.0;

	if (initStateName != NULL) {
	  if (TWOreadState(pDevice, initStateName, 1, &vd, NULL, NULL ) < 0) {
	    fprintf(stderr,
		"NUMD2load: trouble reading state-file %s\n", initStateName);
	  } else {
	    *(ckt->CKTstate0 + inst->NUMD2voltage) = vd;
	    NUMD2setBCs(pDevice, vd);
	    delVd = 0.0;
	  }
	}
      }
      /*
       * compute dc current and derivitives
       */
      /* use the routines for numerical simulation */

      if (ckt->CKTmode & (MODEDCOP | MODETRANOP | MODEDCTRANCURVE | MODEINITSMSIG)) {
	numDevNonCon = 0;
	inst->NUMD2c11 = inst->NUMD2y11r = inst->NUMD2y11i = 0.0;
	inst->NUMD2smSigAvail = FALSE;
    devNonCon:
	NUMD2project(pDevice, delVd);
	if (TWOdcDebug) {
	  printVoltages(stdout,
	      model->NUMD2modName, inst->NUMD2name,
	      deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
	}
	TWObiasSolve(pDevice, MaxIterations, FALSE, model->NUMD2pInfo);
	devConverged = pDevice->converged;

	if (devConverged && finite(pDevice->rhsNorm)) {
	  /* extract the current and conductance information */
	  NUMD2current(pDevice, FALSE, NULL, &id);
	  NUMD2conductance(pDevice, FALSE, NULL, &gd);
	} else {
	  /* do voltage step backtracking */
	  /* restore the boundary nodes to the previous value */
	  NUMD2setBCs(pDevice, vd - delVd);
	  TWOstoreInitialGuess(pDevice);
	  TWOresetJacobian(pDevice);
	  delVd *= 0.5;
	  vd = delVd + *(ckt->CKTstate0 + inst->NUMD2voltage);
	  numDevNonCon++;
	  Check = 1;
	  if (numDevNonCon > 10) {
	    printVoltages(stderr,
		model->NUMD2modName, inst->NUMD2name,
		deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
	    fprintf(stderr,
		"*** Non-convergence during load ***\n");
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_DC] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  } else {
	    goto devNonCon;
	  }
	}
      }
      if ( (ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
	  ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) {
	/*
	 * store small-signal parameters
	 */
	if ((!(ckt->CKTmode & MODETRANOP)) ||
	    (!(ckt->CKTmode & MODEUIC))) {
	  if (ckt->CKTmode & MODEINITSMSIG) {
	    totalTime = SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_DC] += totalTime;
	    startTime2 = SPfrontEnd->IFseconds();
	    NUMD2initSmSig(inst);
	    pDevice->pStats->totalTime[STAT_AC] +=
		SPfrontEnd->IFseconds() - startTime2;
	    continue;
	  } else {
	    inst->NUMD2smSigAvail = FALSE;
	  }
	  /*
	   * transient analysis
	   */
	  if (ckt->CKTmode & MODEINITPRED) {
	    NUMD2setBCs(pDevice, vd);
	    TWOstoreInitialGuess(pDevice);
	  } else {
	    NUMD2update(pDevice, delVd, TRUE);
	  }
	  if (TWOtranDebug) {
	    printVoltages(stdout,
		model->NUMD2modName, inst->NUMD2name,
		deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
	  }
	  TWObiasSolve(pDevice, 0, TRUE, model->NUMD2pInfo);

	  if (!finite(pDevice->rhsNorm)) {
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_TRAN] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  }
	  devConverged = TWOdeviceConverged(pDevice);
	  pDevice->converged = devConverged;

	  /* extract the current and conductance information */
	  NUMD2current(pDevice, TRUE, model->NUMD2pInfo->intCoeff, &id);
	  NUMD2conductance(pDevice, TRUE, model->NUMD2pInfo->intCoeff, &gd);
	}
      }
      /*
       * check convergence
       */
      if ((!(ckt->CKTmode & MODEINITFIX)) || (!(inst->NUMD2off))) {
	if (Check == 1 || !devConverged) {
	  ckt->CKTnoncon++;
	  ckt->CKTtroubleElt = (GENinstance *) inst;
	} else {
	  tol = ckt->CKTreltol * MAX(fabs(idhat), fabs(id)) + ckt->CKTabstol;
	  if (fabs(idhat - id) > tol) {
	    ckt->CKTnoncon++;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	  }
	}
      }
      *(ckt->CKTstate0 + inst->NUMD2voltage) = vd;
      *(ckt->CKTstate0 + inst->NUMD2id) = id;
      *(ckt->CKTstate0 + inst->NUMD2conduct) = gd;

  load:

      /*
       * load current vector
       */
      ideq = id - gd * vd;
      *(ckt->CKTrhs + inst->NUMD2negNode) += ideq;
      *(ckt->CKTrhs + inst->NUMD2posNode) -= ideq;
      /*
       * load matrix
       */
      *(inst->NUMD2posPosPtr) += gd;
      *(inst->NUMD2negNegPtr) += gd;
      *(inst->NUMD2negPosPtr) -= gd;
      *(inst->NUMD2posNegPtr) -= gd;

      totalTime += SPfrontEnd->IFseconds() - startTime;
      if (ckt->CKTmode & MODETRAN) {
	pDevice->pStats->totalTime[STAT_TRAN] += totalTime;
      } else {
	pDevice->pStats->totalTime[STAT_DC] += totalTime;
      }
    }
  }
  return (OK);
}

int
NUMD2initSmSig(NUMD2instance *inst)
{
  SPcomplex yd;
  double omega = NUMD2modPtr(inst)->NUMD2methods->METHomega;

  AcAnalysisMethod = SOR_ONLY;
  (void) NUMD2admittance(inst->NUMD2pDevice, omega, &yd);
  inst->NUMD2c11 = yd.imag / omega;
  inst->NUMD2y11r = yd.real;
  inst->NUMD2y11i = yd.imag;
  inst->NUMD2smSigAvail = TRUE;
  return (OK);
}
