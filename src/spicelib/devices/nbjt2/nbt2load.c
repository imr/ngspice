/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This is the function called each iteration to evaluate the 2d numerical
 * BJTs in the circuit and load them into the matrix as appropriate
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "nbjt2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"

/* External Declarations */

/* Check out this one */
extern int NBJT2initSmSig(NBJT2instance *);




int
NBJT2load(GENmodel *inModel, CKTcircuit *ckt)
{
  register NBJT2model *model = (NBJT2model *) inModel;
  register NBJT2instance *inst;
  register TWOdevice *pDevice;
  double startTime, startTime2, totalTime, totalTime2;
  double tol;
  double ic = 0.0, ie = 0.0;
  double iceq, ieeq;
  double ichat = 0.0, iehat = 0.0;
  double delVce, delVbe;
  double vce, vbe;
  double dIeDVce = 0.0, dIeDVbe = 0.0;
  double dIcDVce = 0.0, dIcDVbe = 0.0;
  double xfact;
  int icheck;
  int icheck1;
  int i;
  double deltaNorm[7];
  int devConverged = 0;
  int numDevNonCon;
  int deviceType;
  int doInitSolve;
  int doVoltPred;
  char *initStateName;

  /* loop through all the models */
  for (; model != NULL; model = NBJT2nextModel(model)) {
    FieldDepMobility = model->NBJT2models->MODLfieldDepMobility;
    TransDepMobility = model->NBJT2models->MODLtransDepMobility;
    SurfaceMobility = model->NBJT2models->MODLsurfaceMobility;
    Srh = model->NBJT2models->MODLsrh;
    Auger = model->NBJT2models->MODLauger;
    AvalancheGen = model->NBJT2models->MODLavalancheGen;
    OneCarrier = model->NBJT2methods->METHoneCarrier;
    MobDeriv = model->NBJT2methods->METHmobDeriv;
    MaxIterations = model->NBJT2methods->METHitLim;
    TWOdcDebug = model->NBJT2outputs->OUTPdcDebug;
    TWOtranDebug = model->NBJT2outputs->OUTPtranDebug;
    TWOacDebug = model->NBJT2outputs->OUTPacDebug;
    deviceType = model->NBJT2options->OPTNdeviceType;
    doVoltPred = model->NBJT2methods->METHvoltPred;

    if (ckt->CKTmode & MODEINITPRED) {
      /* compute normalized deltas and predictor coeff */
      if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	model->NBJT2pInfo->order = ckt->CKTorder;
	model->NBJT2pInfo->method = ckt->CKTintegrateMethod;
	for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	  deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
	}
	computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NBJT2pInfo->intCoeff, deltaNorm);
	computePredCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NBJT2pInfo->predCoeff, deltaNorm);
      }
    } else if (ckt->CKTmode & MODEINITTRAN) {
      model->NBJT2pInfo->order = ckt->CKTorder;
      model->NBJT2pInfo->method = ckt->CKTintegrateMethod;
      for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
      }
      computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	  model->NBJT2pInfo->intCoeff, deltaNorm);
    }
    /* loop through all the instances of the model */
    for (inst = NBJT2instances(model); inst != NULL;
         inst = NBJT2nextInstance(inst)) {

      pDevice = inst->NBJT2pDevice;

      totalTime = 0.0;
      startTime = SPfrontEnd->IFseconds();

      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NBJT2globals));

      /*
       * initialization
       */
      pDevice->devStates = ckt->CKTstates;
      icheck = 1;
      doInitSolve = FALSE;
      initStateName = NULL;
      if (ckt->CKTmode & MODEINITSMSIG) {
	vbe = *(ckt->CKTstate0 + inst->NBJT2vbe);
	vce = *(ckt->CKTstate0 + inst->NBJT2vce);
	delVbe = 0.0;
	delVce = 0.0;
	NBJT2setBCs(pDevice, vce, vbe);
      } else if (ckt->CKTmode & MODEINITTRAN) {
	*(ckt->CKTstate0 + inst->NBJT2vbe) =
	    *(ckt->CKTstate1 + inst->NBJT2vbe);
	*(ckt->CKTstate0 + inst->NBJT2vce) =
	    *(ckt->CKTstate1 + inst->NBJT2vce);
	vbe = *(ckt->CKTstate1 + inst->NBJT2vbe);
	vce = *(ckt->CKTstate1 + inst->NBJT2vce);
	TWOsaveState(pDevice);
	delVbe = 0.0;
	delVce = 0.0;
      } else if ((ckt->CKTmode & MODEINITJCT) &&
	  (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
	doInitSolve = TRUE;
	initStateName = inst->NBJT2icFile;
	vbe = 0.0;
	vce = 0.0;
	delVbe = vbe;
	delVce = vce;
      } else if ((ckt->CKTmode & MODEINITJCT) && (inst->NBJT2off == 0)) {
	doInitSolve = TRUE;
	initStateName = inst->NBJT2icFile;
	if (deviceType == OPTN_JFET) {
	  vbe = 0.0;
	  vce = inst->NBJT2type * 0.5;
	} else {
	  vbe = inst->NBJT2type * 0.6;
	  vce = inst->NBJT2type * 1.0;
	}
	delVbe = vbe;
	delVce = vce;
      } else if (ckt->CKTmode & MODEINITJCT) {
	doInitSolve = TRUE;
	vbe = 0.0;
	vce = 0.0;
	delVbe = vbe;
	delVce = vce;
      } else if ((ckt->CKTmode & MODEINITFIX) && inst->NBJT2off) {
	vbe = 0.0;
	vce = 0.0;
	delVbe = vbe;
	delVce = vce;
      } else {
	if (ckt->CKTmode & MODEINITPRED) {
	  *(ckt->CKTstate0 + inst->NBJT2vbe) =
	      *(ckt->CKTstate1 + inst->NBJT2vbe);
	  *(ckt->CKTstate0 + inst->NBJT2vce) =
	      *(ckt->CKTstate1 + inst->NBJT2vce);
	  *(ckt->CKTstate0 + inst->NBJT2ic) =
	      *(ckt->CKTstate1 + inst->NBJT2ic);
	  *(ckt->CKTstate0 + inst->NBJT2ie) =
	      *(ckt->CKTstate1 + inst->NBJT2ie);
	  *(ckt->CKTstate0 + inst->NBJT2dIeDVce) =
	      *(ckt->CKTstate1 + inst->NBJT2dIeDVce);
	  *(ckt->CKTstate0 + inst->NBJT2dIeDVbe) =
	      *(ckt->CKTstate1 + inst->NBJT2dIeDVbe);
	  *(ckt->CKTstate0 + inst->NBJT2dIcDVce) =
	      *(ckt->CKTstate1 + inst->NBJT2dIcDVce);
	  *(ckt->CKTstate0 + inst->NBJT2dIcDVbe) =
	      *(ckt->CKTstate1 + inst->NBJT2dIcDVbe);
	  /* compute normalized deltas and predictor coeff */
	  if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	    /* no linear prediction on device voltages */
	    vbe = *(ckt->CKTstate1 + inst->NBJT2vbe);
	    vce = *(ckt->CKTstate1 + inst->NBJT2vce);
	    TWOpredict(pDevice, model->NBJT2pInfo);
	  } else {
            if (doVoltPred) {
	      /* linear prediction */
	      xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
	      vbe = (1+xfact) * (*(ckt->CKTstate1 + inst->NBJT2vbe))
		  -   (xfact) * (*(ckt->CKTstate2 + inst->NBJT2vbe));
	      vce = (1+xfact) * (*(ckt->CKTstate1 + inst->NBJT2vce))
		  -   (xfact) * (*(ckt->CKTstate2 + inst->NBJT2vce));
	    } else {
	      vbe = *(ckt->CKTstate1 + inst->NBJT2vbe);
	      vce = *(ckt->CKTstate1 + inst->NBJT2vce);
	    }
	  }
	} else {
	  /*
	   * compute new nonlinear branch voltages
	   */
	  vbe = *(ckt->CKTrhsOld + inst->NBJT2baseNode) -
	      *(ckt->CKTrhsOld + inst->NBJT2emitNode);
	  vce = *(ckt->CKTrhsOld + inst->NBJT2colNode) -
	      *(ckt->CKTrhsOld + inst->NBJT2emitNode);
	}
	delVbe = vbe - *(ckt->CKTstate0 + inst->NBJT2vbe);
	delVce = vce - *(ckt->CKTstate0 + inst->NBJT2vce);
	ichat = *(ckt->CKTstate0 + inst->NBJT2ic) +
	    *(ckt->CKTstate0 + inst->NBJT2dIcDVbe) * delVbe +
	    *(ckt->CKTstate0 + inst->NBJT2dIcDVce) * delVce;
	iehat = *(ckt->CKTstate0 + inst->NBJT2ie) +
	    *(ckt->CKTstate0 + inst->NBJT2dIeDVbe) * delVbe +
	    *(ckt->CKTstate0 + inst->NBJT2dIeDVce) * delVce;


#ifndef NOBYPASS
	/*
	 * bypass if solution has not changed
	 */
	/*
	 * the following collections of if's would be just one if the average
	 * compiler could handle it, but many find the expression too
	 * complicated, thus the split.
	 */
	if ((ckt->CKTbypass) && pDevice->converged &&
	    (!(ckt->CKTmode & MODEINITPRED)) &&
	    (fabs(delVbe) < (ckt->CKTreltol * MAX(fabs(vbe),
			fabs(*(ckt->CKTstate0 + inst->NBJT2vbe))) +
		    ckt->CKTvoltTol)))
	  if ((fabs(delVce) < ckt->CKTreltol * MAX(fabs(vce),
		      fabs(*(ckt->CKTstate0 + inst->NBJT2vce))) +
		  ckt->CKTvoltTol))
	    if ((fabs(ichat - *(ckt->CKTstate0 + inst->NBJT2ic)) <
		    ckt->CKTreltol * MAX(fabs(ichat),
			fabs(*(ckt->CKTstate0 + inst->NBJT2ic))) +
		    ckt->CKTabstol))
	      if ((fabs(iehat - *(ckt->CKTstate0 + inst->NBJT2ie)) <
		      ckt->CKTreltol * MAX(fabs(iehat),
			  fabs(*(ckt->CKTstate0 + inst->NBJT2ie))) +
		      ckt->CKTabstol)) {
		/*
		 * bypassing....
		 */
		vbe = *(ckt->CKTstate0 + inst->NBJT2vbe);
		vce = *(ckt->CKTstate0 + inst->NBJT2vce);
		ic = *(ckt->CKTstate0 + inst->NBJT2ic);
		ie = *(ckt->CKTstate0 + inst->NBJT2ie);
		dIeDVce = *(ckt->CKTstate0 + inst->NBJT2dIeDVce);
		dIeDVbe = *(ckt->CKTstate0 + inst->NBJT2dIeDVbe);
		dIcDVce = *(ckt->CKTstate0 + inst->NBJT2dIcDVce);
		dIcDVbe = *(ckt->CKTstate0 + inst->NBJT2dIcDVbe);
		goto load;
	      }
#endif				/* NOBYPASS */
	/*
	 * limit nonlinear branch voltages
	 */
	icheck1 = 1;
	if (deviceType == OPTN_JFET) {
	  double vbc, vbc0;
	  vbe = inst->NBJT2type * limitJunctionVoltage(inst->NBJT2type * vbe,
	    inst->NBJT2type * *(ckt->CKTstate0 + inst->NBJT2vbe), &icheck);
	  vbc = vbe - vce;
	  vbc0 = *(ckt->CKTstate0 + inst->NBJT2vbe) -
	      *(ckt->CKTstate0 + inst->NBJT2vce);
	  vbc = inst->NBJT2type * limitJunctionVoltage(inst->NBJT2type * vbc,
	    inst->NBJT2type * vbc0, &icheck);
	  if (icheck1 == 1)
	    icheck = 1;
	  vce = vbe - vbc;
	} else {
	  vbe = inst->NBJT2type * limitJunctionVoltage(inst->NBJT2type * vbe,
	    inst->NBJT2type * *(ckt->CKTstate0 + inst->NBJT2vbe), &icheck);
	  vce = inst->NBJT2type * limitVce(inst->NBJT2type * vce,
	    inst->NBJT2type * *(ckt->CKTstate0 + inst->NBJT2vce), &icheck1);
	  if (icheck1 == 1)
	    icheck = 1;
	}
	delVbe = vbe - *(ckt->CKTstate0 + inst->NBJT2vbe);
	delVce = vce - *(ckt->CKTstate0 + inst->NBJT2vce);
	NBJT2setBCs(pDevice, vce - delVce, vbe - delVbe);
      }

      if (doInitSolve) {
	if (TWOdcDebug) {
	  printVoltages(stdout, model->NBJT2modName, inst->NBJT2name,
	      deviceType, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
	startTime2 = SPfrontEnd->IFseconds();
	TWOequilSolve(pDevice);
	totalTime2 = SPfrontEnd->IFseconds() - startTime2;
	pDevice->pStats->totalTime[STAT_SETUP] += totalTime2;
	pDevice->pStats->totalTime[STAT_DC] -= totalTime2;

	TWObiasSolve(pDevice, MaxIterations, FALSE, NULL);

	*(ckt->CKTstate0 + inst->NBJT2vbe) = 0.0;
	*(ckt->CKTstate0 + inst->NBJT2vce) = 0.0;

	if (initStateName != NULL) {
	  if (TWOreadState(pDevice, initStateName, 2, &vce, &vbe, NULL ) < 0) {
	    fprintf(stderr,
		"NBJT2load: trouble reading state-file %s\n", initStateName);
	  } else {
	    NBJT2setBCs(pDevice, vce, vbe);
	    delVce = delVbe = 0.0;
	  }
	}
      }
      /*
       * determine dc current and derivatives using the numerical routines
       */
      if (ckt->CKTmode & (MODEDCOP | MODETRANOP | MODEDCTRANCURVE | MODEINITSMSIG)) {

	numDevNonCon = 0;
	inst->NBJT2c11 = inst->NBJT2y11r = inst->NBJT2y11i = 0.0;
	inst->NBJT2c12 = inst->NBJT2y12r = inst->NBJT2y12i = 0.0;
	inst->NBJT2c21 = inst->NBJT2y21r = inst->NBJT2y21i = 0.0;
	inst->NBJT2c22 = inst->NBJT2y22r = inst->NBJT2y22i = 0.0;
	inst->NBJT2smSigAvail = FALSE;
    devNonCon:
	NBJT2project(pDevice, delVce, delVbe);
	if (TWOdcDebug) {
	  printVoltages(stdout, model->NBJT2modName, inst->NBJT2name,
	      deviceType, 2, vce, delVce, vbe, delVbe, 0.0, 0.0);
	}
	TWObiasSolve(pDevice, MaxIterations, FALSE, model->NBJT2pInfo);

	devConverged = pDevice->converged;
	if (devConverged && finite(pDevice->rhsNorm)) {
	  /* compute the currents */
	  NBJT2current(pDevice, FALSE, NULL, &ie, &ic);
	  NBJT2conductance(pDevice, FALSE, NULL,
	      &dIeDVce, &dIcDVce, &dIeDVbe, &dIcDVbe);

	} else {
	  /* reduce the voltage step until converged */
	  /* restore boundary nodes to previous potential */
	  NBJT2setBCs(pDevice, vce - delVce, vbe - delVbe);
	  TWOstoreInitialGuess(pDevice);
	  TWOresetJacobian(pDevice);
	  delVbe *= 0.5;
	  delVce *= 0.5;
	  vbe = delVbe + *(ckt->CKTstate0 + inst->NBJT2vbe);
	  vce = delVce + *(ckt->CKTstate0 + inst->NBJT2vce);
	  numDevNonCon++;
	  icheck = 1;
	  if (numDevNonCon > 10) {
	    printVoltages(stderr, model->NBJT2modName, inst->NBJT2name,
		deviceType, 2, vce, delVce, vbe, delVbe, 0.0, 0.0);
	    fprintf(stderr, "*** Non-convergence during load ***\n");
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_DC] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  } else {
	    goto devNonCon;
	  }
	}
      }
      if ((ckt->CKTmode & (MODETRAN | MODEAC)) ||
	  ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ||
	  (ckt->CKTmode & MODEINITSMSIG)) {
	/*
	 * store small-signal parameters
	 */
	if ((!(ckt->CKTmode & MODETRANOP)) ||
	    (!(ckt->CKTmode & MODEUIC))) {
	  if (ckt->CKTmode & MODEINITSMSIG) {
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_DC] += totalTime;
	    startTime2 = SPfrontEnd->IFseconds();
	    NBJT2initSmSig(inst);
	    pDevice->pStats->totalTime[STAT_AC] +=
		SPfrontEnd->IFseconds() - startTime2;
	    continue;
	  } else {
	    inst->NBJT2smSigAvail = FALSE;
	  }
	  /*
	   * transient analysis
	   */
	  if (ckt->CKTmode & MODEINITPRED) {
	    NBJT2setBCs(pDevice, vce, vbe);
	    TWOstoreInitialGuess(pDevice);
	  } else {
	    NBJT2update(pDevice, delVce, delVbe, TRUE);
	  }
	  if (TWOtranDebug) {
	    printVoltages(stdout, model->NBJT2modName, inst->NBJT2name,
		deviceType, 2, vce, delVce, vbe, delVbe, 0.0, 0.0);
	  }
	  TWObiasSolve(pDevice, 0, TRUE, model->NBJT2pInfo);
	  if (!finite(pDevice->rhsNorm)) {
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_TRAN] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  }
	  devConverged = TWOdeviceConverged(pDevice);
	  pDevice->converged = devConverged;

	  /* compute the currents */
	  NBJT2current(pDevice, TRUE,
	      model->NBJT2pInfo->intCoeff, &ie, &ic);
	  NBJT2conductance(pDevice, TRUE,
	      model->NBJT2pInfo->intCoeff,
	      &dIeDVce, &dIcDVce, &dIeDVbe, &dIcDVbe);
	}
      }
      /*
       * check convergence
       */
      if ((!(ckt->CKTmode & MODEINITFIX)) || (!(inst->NBJT2off))) {
	if (icheck == 1 || !devConverged) {
	  ckt->CKTnoncon++;
	  ckt->CKTtroubleElt = (GENinstance *) inst;
	} else {
	  tol = ckt->CKTreltol * MAX(fabs(ichat), fabs(ic)) + ckt->CKTabstol;
	  if (fabs(ichat - ic) > tol) {
	    ckt->CKTnoncon++;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	  } else {
	    tol = ckt->CKTreltol * MAX(fabs(iehat), fabs(ie)) +
		ckt->CKTabstol;
	    if (fabs(iehat - ie) > tol) {
	      ckt->CKTnoncon++;
	      ckt->CKTtroubleElt = (GENinstance *) inst;
	    }
	  }
	}
      }
      *(ckt->CKTstate0 + inst->NBJT2vbe) = vbe;
      *(ckt->CKTstate0 + inst->NBJT2vce) = vce;
      *(ckt->CKTstate0 + inst->NBJT2ic) = ic;
      *(ckt->CKTstate0 + inst->NBJT2ie) = ie;
      *(ckt->CKTstate0 + inst->NBJT2dIeDVce) = dIeDVce;
      *(ckt->CKTstate0 + inst->NBJT2dIeDVbe) = dIeDVbe;
      *(ckt->CKTstate0 + inst->NBJT2dIcDVce) = dIcDVce;
      *(ckt->CKTstate0 + inst->NBJT2dIcDVbe) = dIcDVbe;

  load:
      /*
       * load current excitation vector
       */

      iceq = ic - dIcDVce * vce - dIcDVbe * vbe;
      ieeq = ie - dIeDVce * vce - dIeDVbe * vbe;
      *(ckt->CKTrhs + inst->NBJT2colNode) -= iceq;
      *(ckt->CKTrhs + inst->NBJT2baseNode) += ieeq + iceq;
      *(ckt->CKTrhs + inst->NBJT2emitNode) -= ieeq;
      /*
       * load y matrix
       */
      *(inst->NBJT2colColPtr) += dIcDVce;
      *(inst->NBJT2colBasePtr) += dIcDVbe;
      *(inst->NBJT2colEmitPtr) -= dIcDVbe + dIcDVce;
      *(inst->NBJT2baseColPtr) -= dIcDVce + dIeDVce;
      *(inst->NBJT2baseBasePtr) -= dIcDVbe + dIeDVbe;
      *(inst->NBJT2baseEmitPtr) += dIcDVbe + dIcDVce + dIeDVbe + dIeDVce;
      *(inst->NBJT2emitColPtr) += dIeDVce;
      *(inst->NBJT2emitBasePtr) += dIeDVbe;
      *(inst->NBJT2emitEmitPtr) -= dIeDVbe + dIeDVce;

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
NBJT2initSmSig(NBJT2instance *inst)
{
  SPcomplex yIeVce, yIeVbe;
  SPcomplex yIcVce, yIcVbe;
  double omega = NBJT2modPtr(inst)->NBJT2methods->METHomega;

  AcAnalysisMethod = SOR_ONLY;
  (void) NBJT2admittance(inst->NBJT2pDevice, omega,
      &yIeVce, &yIcVce, &yIeVbe, &yIcVbe);
  inst->NBJT2c11 = yIcVce.imag / omega;
  inst->NBJT2c12 = yIcVbe.imag / omega;
  inst->NBJT2c21 = (yIeVce.imag - yIcVce.imag) / omega;
  inst->NBJT2c22 = (yIeVbe.imag - yIcVbe.imag) / omega;
  inst->NBJT2y11r = yIcVce.real;
  inst->NBJT2y11i = yIcVce.imag;
  inst->NBJT2y12r = yIcVbe.real;
  inst->NBJT2y12i = yIcVbe.imag;
  inst->NBJT2y21r = yIeVce.real - yIcVce.real;
  inst->NBJT2y21i = yIeVce.imag - yIcVce.imag;
  inst->NBJT2y22r = yIeVbe.real - yIcVbe.real;
  inst->NBJT2y22i = yIeVbe.imag - yIcVbe.imag;
  inst->NBJT2smSigAvail = TRUE;
  return (OK);
}
