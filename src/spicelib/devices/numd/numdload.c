/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "numddefs.h"
#include "ngspice/numenum.h"
#include "ngspice/trandefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"



/* Forward declarations */
int NUMDinitSmSig(NUMDinstance *);

/* External Declarations */
extern int ONEdcDebug;
extern int ONEtranDebug;
extern int ONEacDebug; 


int
NUMDload(GENmodel *inModel, CKTcircuit *ckt)
{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  register ONEdevice *pDevice;
  double startTime, startTime2, totalTime, totalTime2;
  double tol;			/* temporary for tolerance calculations */
  double id = 0.0;
  double ideq;
  double idhat = 0.0;
  double delVd;
  double vd;			/* current diode voltage */
  double gd = 0.0;
  double xfact;
  int check;
  int i;
  double deltaNorm[7];
  int devConverged = FALSE; 
  int numDevNonCon;
  int deviceType;
  int doInitSolve;
  int doVoltPred;
  char *initStateName;

  /* loop through all the diode models */
  for (; model != NULL; model = NUMDnextModel(model)) {
    /* Do model things */
    FieldDepMobility = model->NUMDmodels->MODLfieldDepMobility;
    Srh = model->NUMDmodels->MODLsrh;
    Auger = model->NUMDmodels->MODLauger;
    AvalancheGen = model->NUMDmodels->MODLavalancheGen;
    MobDeriv = model->NUMDmethods->METHmobDeriv;
    MaxIterations = model->NUMDmethods->METHitLim;
    ONEdcDebug = model->NUMDoutputs->OUTPdcDebug;
    ONEtranDebug = model->NUMDoutputs->OUTPtranDebug;
    ONEacDebug = model->NUMDoutputs->OUTPacDebug;
    deviceType = model->NUMDoptions->OPTNdeviceType;
    doVoltPred = model->NUMDmethods->METHvoltPred;

    if (ckt->CKTmode & MODEINITPRED) {
      /* compute normalized deltas and predictor coeff */
      if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	model->NUMDpInfo->order = ckt->CKTorder;
	model->NUMDpInfo->method = ckt->CKTintegrateMethod;
	for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	  deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
	}
	computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMDpInfo->intCoeff, deltaNorm);
	computePredCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMDpInfo->predCoeff, deltaNorm);
      }
    } else if (ckt->CKTmode & MODEINITTRAN) {
      model->NUMDpInfo->order = ckt->CKTorder;
      model->NUMDpInfo->method = ckt->CKTintegrateMethod;
      for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
      }
      computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	  model->NUMDpInfo->intCoeff, deltaNorm);
    }
    /* Now do instance things */
    for (inst = NUMDinstances(model); inst != NULL;
         inst = NUMDnextInstance(inst)) {

      pDevice = inst->NUMDpDevice;

      totalTime = 0.0;
      startTime = SPfrontEnd->IFseconds();

      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMDglobals));

      /*
       * this routine loads diodes for dc and transient analyses.
       */
      pDevice->devStates = ckt->CKTstates;
      /*
       * initialization
       */
      check = 1;
      doInitSolve = FALSE;
      initStateName = NULL;
      if (ckt->CKTmode & MODEINITSMSIG) {
	vd = *(ckt->CKTstate0 + inst->NUMDvoltage);
	delVd = 0.0;
	NUMDsetBCs(pDevice, vd);
      } else if (ckt->CKTmode & MODEINITTRAN) {
	*(ckt->CKTstate0 + inst->NUMDvoltage) =
	    *(ckt->CKTstate1 + inst->NUMDvoltage);
	vd = *(ckt->CKTstate1 + inst->NUMDvoltage);
	ONEsaveState(pDevice);
	delVd = 0.0;
      } else if ((ckt->CKTmode & MODEINITJCT) &&
	  (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
	doInitSolve = TRUE;
	initStateName = inst->NUMDicFile;
	vd = 0.0;
	delVd = vd;
      } else if ((ckt->CKTmode & MODEINITJCT) && inst->NUMDoff) {
	doInitSolve = TRUE;
	vd = 0.0;
	delVd = vd;
      } else if (ckt->CKTmode & MODEINITJCT) {
	doInitSolve = TRUE;
	initStateName = inst->NUMDicFile;
	if (deviceType == OPTN_DIODE) {
	  vd = inst->NUMDtype * 0.5;
	} else if (deviceType == OPTN_MOSCAP) {
	  vd = inst->NUMDtype * 0.8;
	} else {
	  vd = 0.0;
	}
	delVd = vd;
      } else if (ckt->CKTmode & MODEINITFIX && inst->NUMDoff) {
	vd = 0.0;
	delVd = vd;
      } else {
	if (ckt->CKTmode & MODEINITPRED) {
	  *(ckt->CKTstate0 + inst->NUMDvoltage) =
	      *(ckt->CKTstate1 + inst->NUMDvoltage);
	  *(ckt->CKTstate0 + inst->NUMDid) =
	      *(ckt->CKTstate1 + inst->NUMDid);
	  *(ckt->CKTstate0 + inst->NUMDconduct) =
	      *(ckt->CKTstate1 + inst->NUMDconduct);
	  /* compute the normalized deltas */
	  if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	    /* no linear prediction on device voltages */
	    vd = *(ckt->CKTstate1 + inst->NUMDvoltage);
	    ONEpredict(pDevice, model->NUMDpInfo);
	  } else {
            if (doVoltPred) {
	      /* linear prediction */
	      xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
	      vd = (1+xfact) * (*(ckt->CKTstate1 + inst->NUMDvoltage))
		  -  (xfact) * (*(ckt->CKTstate2 + inst->NUMDvoltage));
	    } else {
	      vd = *(ckt->CKTstate1 + inst->NUMDvoltage);
	    }
	  }
	} else {
	  vd = *(ckt->CKTrhsOld + inst->NUMDposNode) -
	      *(ckt->CKTrhsOld + inst->NUMDnegNode);
	}
	delVd = vd - *(ckt->CKTstate0 + inst->NUMDvoltage);
	idhat = *(ckt->CKTstate0 + inst->NUMDid) +
	    *(ckt->CKTstate0 + inst->NUMDconduct) * delVd;  
	/*
	 * bypass if solution has not changed
	 */
	if ((ckt->CKTbypass) && pDevice->converged &&
	    !(ckt->CKTmode & MODEINITPRED)) {
	  tol = ckt->CKTvoltTol + ckt->CKTreltol *
	      MAX(fabs(vd), fabs(*(ckt->CKTstate0 + inst->NUMDvoltage)));
	  if (fabs(delVd) < tol) {
	    tol = ckt->CKTreltol *
		MAX(fabs(idhat), fabs(*(ckt->CKTstate0 + inst->NUMDid))) +
		ckt->CKTabstol;
	    if (fabs(idhat - *(ckt->CKTstate0 + inst->NUMDid))
		< tol) {
	      vd = *(ckt->CKTstate0 + inst->NUMDvoltage);
	      id = *(ckt->CKTstate0 + inst->NUMDid);
	      gd = *(ckt->CKTstate0 + inst->NUMDconduct);
	      goto load;
	    }
	  }
	}
	/*
	 * limit new junction voltage
	 */
	if (deviceType == OPTN_DIODE) {
	  vd = inst->NUMDtype * limitJunctionVoltage(inst->NUMDtype * vd,
	      inst->NUMDtype * *(ckt->CKTstate0 + inst->NUMDvoltage), &check);
	} else if (deviceType == OPTN_MOSCAP) {
	  vd = inst->NUMDtype * limitVgb(inst->NUMDtype * vd,
	      inst->NUMDtype * *(ckt->CKTstate0 + inst->NUMDvoltage), &check);
	} else {
	  vd = inst->NUMDtype * limitResistorVoltage(inst->NUMDtype * vd,
	      inst->NUMDtype * *(ckt->CKTstate0 + inst->NUMDvoltage), &check);
	}
	delVd = vd - *(ckt->CKTstate0 + inst->NUMDvoltage);
      }
      if (doInitSolve) {
	if (ONEdcDebug) {
	  printVoltages(stdout, model->NUMDmodName, inst->NUMDname,
	      deviceType, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
	startTime2 = SPfrontEnd->IFseconds();
	ONEequilSolve(pDevice);
	totalTime2 = SPfrontEnd->IFseconds() - startTime2;
	pDevice->pStats->totalTime[STAT_SETUP] += totalTime2;
	pDevice->pStats->totalTime[STAT_DC] -= totalTime2;

	ONEbiasSolve(pDevice, MaxIterations, FALSE, NULL);

	*(ckt->CKTstate0 + inst->NUMDvoltage) = 0.0;

	if (initStateName != NULL) {
	  if (ONEreadState(pDevice, initStateName, 1, &vd, NULL ) < 0) {
	    fprintf(stderr,
		"NUMDload: trouble reading state-file %s\n", initStateName);
	  } else {
	    NUMDsetBCs(pDevice, vd);
	    delVd = 0.0;
	  }
	}
      }
      /*
       * compute dc current and derivatives
       */
      /* use the routines for numerical simulation */

      if (ckt->CKTmode & (MODEDCOP | MODETRANOP | MODEDCTRANCURVE | MODEINITSMSIG)) {
	numDevNonCon = 0;
	inst->NUMDc11 = inst->NUMDy11r = inst->NUMDy11i = 0.0;
	inst->NUMDsmSigAvail = FALSE;
    devNonCon:
	NUMDproject(pDevice, delVd);
	if (ONEdcDebug) {
	  printVoltages(stdout, model->NUMDmodName, inst->NUMDname,
	      deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
	}
	ONEbiasSolve(pDevice, MaxIterations, FALSE, NULL);
	devConverged = pDevice->converged;
	if (devConverged && finite(pDevice->rhsNorm)) {
	  /* Get the current and conductance information. */
	  NUMDcurrent(pDevice, FALSE, NULL, &id);
	  NUMDconductance(pDevice, FALSE, NULL, &gd);
	} else {
	  /* reduce the voltage step until converged */
	  /* restore the boundary potential to previous value */
	  NUMDsetBCs(pDevice, vd - delVd);
	  ONEstoreInitialGuess(pDevice);
	  ONEresetJacobian(pDevice);
	  delVd *= 0.5;
	  vd = delVd + *(ckt->CKTstate0 + inst->NUMDvoltage);
	  numDevNonCon++;
	  check = 1;
	  if (numDevNonCon > 10) {
	    printVoltages(stderr, model->NUMDmodName, inst->NUMDname,
		deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
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
      if ( (ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
	  ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) {
	/*
	 * store small-signal parameters
	 */
	if ((!(ckt->CKTmode & MODETRANOP)) || (!(ckt->CKTmode & MODEUIC))) {
	  if (ckt->CKTmode & MODEINITSMSIG) {
	    totalTime = SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_DC] += totalTime;
	    startTime2 = SPfrontEnd->IFseconds();
	    NUMDinitSmSig(inst);
	    pDevice->pStats->totalTime[STAT_AC] +=
		SPfrontEnd->IFseconds() - startTime2;
	    continue;
	  } else {
	    inst->NUMDsmSigAvail = FALSE;
	  }
	  /*
	   * transient analysis
	   */
	  if (ckt->CKTmode & MODEINITPRED) {
	    NUMDsetBCs(pDevice, vd);
	    ONEstoreInitialGuess(pDevice);
	  } else {
	    NUMDupdate(pDevice, delVd, TRUE);
	  }
	  if (ONEtranDebug) {
	    printVoltages(stdout, model->NUMDmodName, inst->NUMDname,
		deviceType, 1, vd, delVd, 0.0, 0.0, 0.0, 0.0);
	  }
	  
	  ONEbiasSolve(pDevice, 0, TRUE, model->NUMDpInfo);
	  
	  if (!finite(pDevice->rhsNorm)) {
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_TRAN] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  }
	  
	   pDevice->converged = devConverged = ONEdeviceConverged(pDevice);
             
	  /* extract the current and conductance information */
	  NUMDcurrent(pDevice, TRUE, model->NUMDpInfo->intCoeff, &id);
	  NUMDconductance(pDevice, TRUE, model->NUMDpInfo->intCoeff, &gd);
	}
      }
      /*
       * check convergence
       */
      if ((!(ckt->CKTmode & MODEINITFIX)) || (!(inst->NUMDoff))) {
	if (check == 1 || !devConverged) {
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
      *(ckt->CKTstate0 + inst->NUMDvoltage) = vd;
      *(ckt->CKTstate0 + inst->NUMDid) = id;
      *(ckt->CKTstate0 + inst->NUMDconduct) = gd;

  load:

      /*
       * load current vector
       */
       
      ideq = id - gd * vd;
      *(ckt->CKTrhs + inst->NUMDnegNode) += ideq;
      *(ckt->CKTrhs + inst->NUMDposNode) -= ideq;
   
      /*
       * load matrix
       */
      *(inst->NUMDposPosPtr) += gd;
      *(inst->NUMDnegNegPtr) += gd;
      *(inst->NUMDnegPosPtr) -= gd;
      *(inst->NUMDposNegPtr) -= gd;
      
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
NUMDinitSmSig(NUMDinstance *inst)
{
  SPcomplex yd;
  double omega = NUMDmodPtr(inst)->NUMDmethods->METHomega;

  AcAnalysisMethod = SOR_ONLY;
  (void) NUMDadmittance(inst->NUMDpDevice, omega, &yd);
  inst->NUMDc11 = yd.imag / omega;
  inst->NUMDy11r = yd.real;
  inst->NUMDy11i = yd.imag;
  inst->NUMDsmSigAvail = TRUE;
  return (OK);
}
