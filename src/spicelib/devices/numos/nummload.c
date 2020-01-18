/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This is the function called each iteration to evaluate the 2d numerical
 * MOSFETs in the circuit and load them into the matrix as appropriate
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "numosdef.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"



/* Forward Declarations */

int NUMOSinitSmSig(NUMOSinstance *inst);

int
NUMOSload(GENmodel *inModel, CKTcircuit *ckt)
{
  register NUMOSmodel *model = (NUMOSmodel *) inModel;
  register NUMOSinstance *inst;
  register TWOdevice *pDevice;
  double startTime, startTime2, totalTime, totalTime2;
  double tol;
  double xfact;
  double id = 0.0, is = 0.0, ig = 0.0;
  double ideq, iseq, igeq;
  double idhat = 0.0, ishat = 0.0, ighat = 0.0;
  double delVdb, delVsb, delVgb;
  double vdb, vsb, vgb;
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

  struct mosConductances g;
  /* remove compiler warning */
  g.dIdDVdb = 0.0;
  g.dIdDVsb = 0.0;
  g.dIdDVgb = 0.0;
  g.dIsDVdb = 0.0;
  g.dIsDVsb = 0.0;
  g.dIsDVgb = 0.0;
  g.dIgDVdb = 0.0;
  g.dIgDVsb = 0.0;
  g.dIgDVgb = 0.0;
  /* loop through all the models */
  for (; model != NULL; model = NUMOSnextModel(model)) {
    FieldDepMobility = model->NUMOSmodels->MODLfieldDepMobility;
    TransDepMobility = model->NUMOSmodels->MODLtransDepMobility;
    SurfaceMobility = model->NUMOSmodels->MODLsurfaceMobility;
    Srh = model->NUMOSmodels->MODLsrh;
    Auger = model->NUMOSmodels->MODLauger;
    AvalancheGen = model->NUMOSmodels->MODLavalancheGen;
    OneCarrier = model->NUMOSmethods->METHoneCarrier;
    MobDeriv = model->NUMOSmethods->METHmobDeriv;
    MaxIterations = model->NUMOSmethods->METHitLim;
    TWOdcDebug = model->NUMOSoutputs->OUTPdcDebug;
    TWOtranDebug = model->NUMOSoutputs->OUTPtranDebug;
    TWOacDebug = model->NUMOSoutputs->OUTPacDebug;
    deviceType = model->NUMOSoptions->OPTNdeviceType;
    doVoltPred = model->NUMOSmethods->METHvoltPred;

    if (ckt->CKTmode & MODEINITPRED) {
      /* compute normalized deltas and predictor coeff */
      if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	model->NUMOSpInfo->order = ckt->CKTorder;
	model->NUMOSpInfo->method = ckt->CKTintegrateMethod;
	for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	  deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
	}
	computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMOSpInfo->intCoeff, deltaNorm);
	computePredCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	    model->NUMOSpInfo->predCoeff, deltaNorm);
      }
    } else if (ckt->CKTmode & MODEINITTRAN) {
      model->NUMOSpInfo->order = ckt->CKTorder;
      model->NUMOSpInfo->method = ckt->CKTintegrateMethod;
      for (i = 0; i <= ckt->CKTmaxOrder; i++) {
	deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
      }
      computeIntegCoeff(ckt->CKTintegrateMethod, ckt->CKTorder,
	  model->NUMOSpInfo->intCoeff, deltaNorm);
    }
    /* loop through all the instances of the model */
    for (inst = NUMOSinstances(model); inst != NULL;
         inst = NUMOSnextInstance(inst)) {

      pDevice = inst->NUMOSpDevice;

      totalTime = 0.0;
      startTime = SPfrontEnd->IFseconds();

      /* Get Temp.-Dep. Global Parameters */
      GLOBgetGlobals(&(inst->NUMOSglobals));

      /*
       * initialization
       */
      pDevice->devStates = ckt->CKTstates;
      icheck = 1;
      doInitSolve = FALSE;
      initStateName = NULL;
      if (ckt->CKTmode & MODEINITSMSIG) {
	vdb = *(ckt->CKTstate0 + inst->NUMOSvdb);
	vsb = *(ckt->CKTstate0 + inst->NUMOSvsb);
	vgb = *(ckt->CKTstate0 + inst->NUMOSvgb);
	delVdb = 0.0;
	delVsb = 0.0;
	delVgb = 0.0;
	NUMOSsetBCs(pDevice, vdb, vsb, vgb);
      } else if (ckt->CKTmode & MODEINITTRAN) {
	*(ckt->CKTstate0 + inst->NUMOSvdb) =
	    *(ckt->CKTstate1 + inst->NUMOSvdb);
	*(ckt->CKTstate0 + inst->NUMOSvsb) =
	    *(ckt->CKTstate1 + inst->NUMOSvsb);
	*(ckt->CKTstate0 + inst->NUMOSvgb) =
	    *(ckt->CKTstate1 + inst->NUMOSvgb);
	vdb = *(ckt->CKTstate1 + inst->NUMOSvdb);
	vsb = *(ckt->CKTstate1 + inst->NUMOSvsb);
	vgb = *(ckt->CKTstate1 + inst->NUMOSvgb);
	TWOsaveState(pDevice);
	delVdb = 0.0;
	delVsb = 0.0;
	delVgb = 0.0;
      } else if ((ckt->CKTmode & MODEINITJCT) &&
	  (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
	doInitSolve = TRUE;
	initStateName = inst->NUMOSicFile;
	vdb = 0.0;
	vsb = 0.0;
	vgb = 0.0;
	delVdb = vdb;
	delVsb = vsb;
	delVgb = vgb;
      } else if ((ckt->CKTmode & MODEINITJCT) && (inst->NUMOSoff == 0)) {
	doInitSolve = TRUE;
	initStateName = inst->NUMOSicFile;
	if (deviceType == OPTN_BIPOLAR) {
	  /* d,g,s,b => c,b,e,s */
	  vdb = 0.0;
	  vsb = -inst->NUMOStype * 1.0;
	  vgb = -inst->NUMOStype * (1.0 - 0.6);
	} else if (deviceType == OPTN_JFET) {
	  vdb = inst->NUMOStype * 0.5;
	  vsb = 0.0;
	  vgb = 0.0;
	} else {
	  vdb = inst->NUMOStype * 0.5;
	  vsb = 0.0;
	  vgb = inst->NUMOStype * 1.0;
	}
	delVdb = vdb;
	delVsb = vsb;
	delVgb = vgb;
      } else if (ckt->CKTmode & MODEINITJCT) {
	doInitSolve = TRUE;
	vdb = 0.0;
	vsb = 0.0;
	vgb = 0.0;
	delVdb = vdb;
	delVsb = vsb;
	delVgb = vgb;
      } else if ((ckt->CKTmode & MODEINITFIX) && inst->NUMOSoff) {
	vdb = 0.0;
	vsb = 0.0;
	vgb = 0.0;
	delVdb = vdb;
	delVsb = vsb;
	delVgb = vgb;
      } else {
	if (ckt->CKTmode & MODEINITPRED) {
	  *(ckt->CKTstate0 + inst->NUMOSvdb) =
	      *(ckt->CKTstate1 + inst->NUMOSvdb);
	  *(ckt->CKTstate0 + inst->NUMOSvsb) =
	      *(ckt->CKTstate1 + inst->NUMOSvsb);
	  *(ckt->CKTstate0 + inst->NUMOSvgb) =
	      *(ckt->CKTstate1 + inst->NUMOSvgb);
	  *(ckt->CKTstate0 + inst->NUMOSid) =
	      *(ckt->CKTstate1 + inst->NUMOSid);
	  *(ckt->CKTstate0 + inst->NUMOSis) =
	      *(ckt->CKTstate1 + inst->NUMOSis);
	  *(ckt->CKTstate0 + inst->NUMOSig) =
	      *(ckt->CKTstate1 + inst->NUMOSig);
	  *(ckt->CKTstate0 + inst->NUMOSdIdDVdb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIdDVdb);
	  *(ckt->CKTstate0 + inst->NUMOSdIdDVsb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIdDVsb);
	  *(ckt->CKTstate0 + inst->NUMOSdIdDVgb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIdDVgb);
	  *(ckt->CKTstate0 + inst->NUMOSdIsDVdb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIsDVdb);
	  *(ckt->CKTstate0 + inst->NUMOSdIsDVsb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIsDVsb);
	  *(ckt->CKTstate0 + inst->NUMOSdIsDVgb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIsDVgb);
	  *(ckt->CKTstate0 + inst->NUMOSdIgDVdb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIgDVdb);
	  *(ckt->CKTstate0 + inst->NUMOSdIgDVsb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIgDVsb);
	  *(ckt->CKTstate0 + inst->NUMOSdIgDVgb) =
	      *(ckt->CKTstate1 + inst->NUMOSdIgDVgb);
	  if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
	    /* no linear prediction on device voltages */
	    vdb = *(ckt->CKTstate1 + inst->NUMOSvdb);
	    vsb = *(ckt->CKTstate1 + inst->NUMOSvsb);
	    vgb = *(ckt->CKTstate1 + inst->NUMOSvgb);
	    TWOpredict(pDevice, model->NUMOSpInfo);
	  } else {
            if (doVoltPred) {
	      /* linear prediction */
	      xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
	      vdb = (1+xfact) * (*(ckt->CKTstate1 + inst->NUMOSvdb))
		  -   (xfact) * (*(ckt->CKTstate2 + inst->NUMOSvdb));
	      vsb = (1+xfact) * (*(ckt->CKTstate1 + inst->NUMOSvsb))
		  -   (xfact) * (*(ckt->CKTstate2 + inst->NUMOSvsb));
	      vgb = (1+xfact) * (*(ckt->CKTstate1 + inst->NUMOSvgb))
		  -   (xfact) * (*(ckt->CKTstate2 + inst->NUMOSvgb));
	    } else {
	      vdb = *(ckt->CKTstate1 + inst->NUMOSvdb);
	      vsb = *(ckt->CKTstate1 + inst->NUMOSvsb);
	      vgb = *(ckt->CKTstate1 + inst->NUMOSvgb);
	    }
	  }
	} else {
	  /*
	   * compute new nonlinear branch voltages
	   */
	  vdb = *(ckt->CKTrhsOld + inst->NUMOSdrainNode)
	      - *(ckt->CKTrhsOld + inst->NUMOSbulkNode);
	  vsb = *(ckt->CKTrhsOld + inst->NUMOSsourceNode)
	      - *(ckt->CKTrhsOld + inst->NUMOSbulkNode);
	  vgb = *(ckt->CKTrhsOld + inst->NUMOSgateNode)
	      - *(ckt->CKTrhsOld + inst->NUMOSbulkNode);
	}
	delVdb = vdb - *(ckt->CKTstate0 + inst->NUMOSvdb);
	delVsb = vsb - *(ckt->CKTstate0 + inst->NUMOSvsb);
	delVgb = vgb - *(ckt->CKTstate0 + inst->NUMOSvgb);
	idhat = *(ckt->CKTstate0 + inst->NUMOSid)
	    + *(ckt->CKTstate0 + inst->NUMOSdIdDVdb) * delVdb
	    + *(ckt->CKTstate0 + inst->NUMOSdIdDVsb) * delVsb
	    + *(ckt->CKTstate0 + inst->NUMOSdIdDVgb) * delVgb;
	ishat = *(ckt->CKTstate0 + inst->NUMOSis)
	    + *(ckt->CKTstate0 + inst->NUMOSdIsDVdb) * delVdb
	    + *(ckt->CKTstate0 + inst->NUMOSdIsDVsb) * delVsb
	    + *(ckt->CKTstate0 + inst->NUMOSdIsDVgb) * delVgb;
	ighat = *(ckt->CKTstate0 + inst->NUMOSig)
	    + *(ckt->CKTstate0 + inst->NUMOSdIgDVdb) * delVdb
	    + *(ckt->CKTstate0 + inst->NUMOSdIgDVsb) * delVsb
	    + *(ckt->CKTstate0 + inst->NUMOSdIgDVgb) * delVgb;


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
	    (fabs(delVdb) < (ckt->CKTreltol * MAX(fabs(vdb),
			fabs(*(ckt->CKTstate0 + inst->NUMOSvdb))) +
		    ckt->CKTvoltTol)))
	  if ((fabs(delVsb) < ckt->CKTreltol * MAX(fabs(vsb),
		      fabs(*(ckt->CKTstate0 + inst->NUMOSvsb))) +
		  ckt->CKTvoltTol))
	    if ((fabs(delVgb) < ckt->CKTreltol * MAX(fabs(vgb),
			fabs(*(ckt->CKTstate0 + inst->NUMOSvgb))) +
		    ckt->CKTvoltTol))
	      if ((fabs(idhat - *(ckt->CKTstate0 + inst->NUMOSid)) <
		      ckt->CKTreltol * MAX(fabs(idhat),
			  fabs(*(ckt->CKTstate0 + inst->NUMOSid))) +
		      ckt->CKTabstol))
		if ((fabs(ishat - *(ckt->CKTstate0 + inst->NUMOSis)) <
			ckt->CKTreltol * MAX(fabs(ishat),
			    fabs(*(ckt->CKTstate0 + inst->NUMOSis))) +
			ckt->CKTabstol))
		  if ((fabs(ighat - *(ckt->CKTstate0 + inst->NUMOSig)) <
			  ckt->CKTreltol * MAX(fabs(ighat),
			      fabs(*(ckt->CKTstate0 + inst->NUMOSig))) +
			  ckt->CKTabstol)) {
		    /*
		     * bypassing....
		     */
		    vdb = *(ckt->CKTstate0 + inst->NUMOSvdb);
		    vsb = *(ckt->CKTstate0 + inst->NUMOSvsb);
		    vgb = *(ckt->CKTstate0 + inst->NUMOSvgb);
		    id = *(ckt->CKTstate0 + inst->NUMOSid);
		    is = *(ckt->CKTstate0 + inst->NUMOSis);
		    ig = *(ckt->CKTstate0 + inst->NUMOSig);
		    g.dIdDVdb = *(ckt->CKTstate0 + inst->NUMOSdIdDVdb);
		    g.dIdDVsb = *(ckt->CKTstate0 + inst->NUMOSdIdDVsb);
		    g.dIdDVgb = *(ckt->CKTstate0 + inst->NUMOSdIdDVgb);
		    g.dIsDVdb = *(ckt->CKTstate0 + inst->NUMOSdIsDVdb);
		    g.dIsDVsb = *(ckt->CKTstate0 + inst->NUMOSdIsDVsb);
		    g.dIsDVgb = *(ckt->CKTstate0 + inst->NUMOSdIsDVgb);
		    g.dIgDVdb = *(ckt->CKTstate0 + inst->NUMOSdIgDVdb);
		    g.dIgDVsb = *(ckt->CKTstate0 + inst->NUMOSdIgDVsb);
		    g.dIgDVgb = *(ckt->CKTstate0 + inst->NUMOSdIgDVgb);
		    goto load;
		  }
#endif				/* NOBYPASS */
	/*
	 * limit nonlinear branch voltages
	 */
	icheck1 = 1;
	if (deviceType == OPTN_BIPOLAR) {
	  double vbe, vbe0;
	  double vce, vce0;

	  vdb = -inst->NUMOStype * limitJunctionVoltage(-inst->NUMOStype * vdb,
	      -inst->NUMOStype * *(ckt->CKTstate0 + inst->NUMOSvdb), &icheck);
	  vce = vdb - vsb;
	  vce0 = *(ckt->CKTstate0 + inst->NUMOSvdb) -
	      *(ckt->CKTstate0 + inst->NUMOSvsb);
	  vce = inst->NUMOStype * limitVce(inst->NUMOStype * vce,
	      inst->NUMOStype * vce0, &icheck1);
	  if (icheck1 == 1)
	    icheck = 1;
	  vsb = vdb - vce;
	  vbe = vgb - vsb;
	  vbe0 = *(ckt->CKTstate0 + inst->NUMOSvgb) -
	      *(ckt->CKTstate0 + inst->NUMOSvsb);
	  vbe = inst->NUMOStype * limitVbe(inst->NUMOStype * vbe,
	      inst->NUMOStype * vbe0, &icheck1);
	  if (icheck1 == 1)
	    icheck = 1;
	  vgb = vbe + vsb;
	} else {
	  vdb = -inst->NUMOStype * limitJunctionVoltage(-inst->NUMOStype * vdb,
	      -inst->NUMOStype * *(ckt->CKTstate0 + inst->NUMOSvdb), &icheck);
	  vsb = -inst->NUMOStype * limitJunctionVoltage(-inst->NUMOStype * vsb,
	      -inst->NUMOStype * *(ckt->CKTstate0 + inst->NUMOSvsb), &icheck1);
	  if (icheck1 == 1)
	    icheck = 1;
	  vgb = inst->NUMOStype * limitVgb(inst->NUMOStype * vgb,
	      inst->NUMOStype * *(ckt->CKTstate0 + inst->NUMOSvgb), &icheck1);
	  if (icheck1 == 1)
	    icheck = 1;
	}
	delVdb = vdb - *(ckt->CKTstate0 + inst->NUMOSvdb);
	delVsb = vsb - *(ckt->CKTstate0 + inst->NUMOSvsb);
	delVgb = vgb - *(ckt->CKTstate0 + inst->NUMOSvgb);
	NUMOSsetBCs(pDevice, vdb - delVdb, vsb - delVsb, vgb - delVgb);
      }

      if (doInitSolve) {
	if (TWOdcDebug) {
	  printVoltages(stdout, model->NUMOSmodName, inst->NUMOSname,
	      deviceType, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
	startTime2 = SPfrontEnd->IFseconds();
	TWOequilSolve(pDevice);
	totalTime2 = SPfrontEnd->IFseconds() - startTime2;
	pDevice->pStats->totalTime[STAT_SETUP] += totalTime2;
	pDevice->pStats->totalTime[STAT_DC] -= totalTime2;

	TWObiasSolve(pDevice, MaxIterations, FALSE, NULL);

	*(ckt->CKTstate0 + inst->NUMOSvdb) = 0.0;
	*(ckt->CKTstate0 + inst->NUMOSvsb) = 0.0;
	*(ckt->CKTstate0 + inst->NUMOSvgb) = 0.0;

	if (initStateName != NULL) {
	  if (TWOreadState(pDevice, initStateName, 3, &vdb, &vgb, &vsb ) < 0) {
	    fprintf(stderr,
		"NUMOSload: trouble reading state-file %s\n", initStateName);
	  } else {
	    NUMOSsetBCs(pDevice, vdb, vsb, vgb);
	    delVdb = delVsb = delVgb = 0.0;
	  }
	}
      }
      /*
       * determine dc current and derivatives using the numerical routines
       */
      if (ckt->CKTmode & (MODEDCOP | MODETRANOP | MODEDCTRANCURVE | MODEINITSMSIG)) {
	numDevNonCon = 0;
	inst->NUMOSc11 = inst->NUMOSy11r = inst->NUMOSy11i = 0.0;
	inst->NUMOSc12 = inst->NUMOSy12r = inst->NUMOSy12i = 0.0;
	inst->NUMOSc13 = inst->NUMOSy13r = inst->NUMOSy13i = 0.0;
	inst->NUMOSc21 = inst->NUMOSy21r = inst->NUMOSy21i = 0.0;
	inst->NUMOSc22 = inst->NUMOSy22r = inst->NUMOSy22i = 0.0;
	inst->NUMOSc23 = inst->NUMOSy23r = inst->NUMOSy23i = 0.0;
	inst->NUMOSc31 = inst->NUMOSy31r = inst->NUMOSy31i = 0.0;
	inst->NUMOSc32 = inst->NUMOSy32r = inst->NUMOSy32i = 0.0;
	inst->NUMOSc33 = inst->NUMOSy33r = inst->NUMOSy33i = 0.0;
	inst->NUMOSsmSigAvail = FALSE;
    devNonCon:
	NUMOSproject(pDevice, delVdb, delVsb, delVgb);
	if (TWOdcDebug) {
	  printVoltages(stdout, model->NUMOSmodName, inst->NUMOSname,
	      deviceType, 3, vdb, delVdb, vgb, delVgb, vsb, delVsb);
	}
	TWObiasSolve(pDevice, MaxIterations, FALSE, model->NUMOSpInfo);

	devConverged = pDevice->converged;
	if (devConverged && finite(pDevice->rhsNorm)) {
	  /* compute the currents */
	  NUMOScurrent(pDevice, FALSE, NULL, &id, &is, &ig);
	  NUMOSconductance(pDevice, FALSE, NULL, &g);
	  /*
	   * Add gmin to the gate conductance terms since they will be zero.
	   * XXX This messes up the gXY output values, but we choose not to
	   * correct this error, because it shouldn't cause practical problems.
	   */
	  g.dIgDVdb += ckt->CKTgmin;
	  g.dIgDVsb += ckt->CKTgmin;
	  g.dIgDVgb += ckt->CKTgmin;

	} else {
	  /* reduce the voltage step until converged */
	  /* restore boundary nodes to previous potential */
	  NUMOSsetBCs(pDevice,
	      vdb - delVdb, vsb - delVsb, vgb - delVgb);
	  TWOstoreInitialGuess(pDevice);
	  TWOresetJacobian(pDevice);
	  delVdb *= 0.5;
	  delVsb *= 0.5;
	  delVgb *= 0.5;
	  vdb = delVdb + *(ckt->CKTstate0 + inst->NUMOSvdb);
	  vsb = delVsb + *(ckt->CKTstate0 + inst->NUMOSvsb);
	  vgb = delVgb + *(ckt->CKTstate0 + inst->NUMOSvgb);
	  numDevNonCon++;
	  icheck = 1;
	  if (numDevNonCon > 10) {
	    printVoltages(stderr, model->NUMOSmodName, inst->NUMOSname,
		deviceType, 3, vdb, delVdb, vgb, delVgb, vsb, delVsb);
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
	    NUMOSinitSmSig(inst);
	    pDevice->pStats->totalTime[STAT_AC] +=
		SPfrontEnd->IFseconds() - startTime2;
	    continue;
	  } else {
	    inst->NUMOSsmSigAvail = FALSE;
	  }
	  /*
	   * transient analysis
	   */
	  if (ckt->CKTmode & MODEINITPRED) {
	    NUMOSsetBCs(pDevice, vdb, vsb, vgb);
	    TWOstoreInitialGuess(pDevice);
	  } else {
	    NUMOSupdate(pDevice, delVdb, delVsb, delVgb, TRUE);
	  }
	  if (TWOtranDebug) {
	    printVoltages(stdout, model->NUMOSmodName, inst->NUMOSname,
		deviceType, 3, vdb, delVdb, vgb, delVgb, vsb, delVsb);
	  }
	  TWObiasSolve(pDevice, 0, TRUE, model->NUMOSpInfo);
	  if (!finite(pDevice->rhsNorm)) {
	    totalTime += SPfrontEnd->IFseconds() - startTime;
	    pDevice->pStats->totalTime[STAT_TRAN] += totalTime;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	    return (E_BADMATRIX);
	  }
	  devConverged = TWOdeviceConverged(pDevice);
	  pDevice->converged = devConverged;

	  /* compute the currents and conductances */
	  NUMOScurrent(pDevice, TRUE, model->NUMOSpInfo->intCoeff,
	      &id, &is, &ig);
	  NUMOSconductance(pDevice, TRUE,
	      model->NUMOSpInfo->intCoeff, &g);
	}
      }
      /*
       * check convergence
       */
      if ((!(ckt->CKTmode & MODEINITFIX)) || (!(inst->NUMOSoff))) {
	if (icheck == 1 || !devConverged) {
	  ckt->CKTnoncon++;
	  ckt->CKTtroubleElt = (GENinstance *) inst;
	} else {
	  tol = ckt->CKTreltol * MAX(fabs(idhat), fabs(id)) + ckt->CKTabstol;
	  if (fabs(idhat - id) > tol) {
	    ckt->CKTnoncon++;
	    ckt->CKTtroubleElt = (GENinstance *) inst;
	  } else {
	    tol = ckt->CKTreltol * MAX(fabs(ishat), fabs(is)) +
		ckt->CKTabstol;
	    if (fabs(ishat - is) > tol) {
	      ckt->CKTnoncon++;
	      ckt->CKTtroubleElt = (GENinstance *) inst;
	    } else {
	      tol = ckt->CKTreltol * MAX(fabs(ighat), fabs(ig)) +
		  ckt->CKTabstol;
	      if (fabs(ighat - ig) > tol) {
		ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) inst;
	      }
	    }
	  }
	}
      }
      *(ckt->CKTstate0 + inst->NUMOSvdb) = vdb;
      *(ckt->CKTstate0 + inst->NUMOSvsb) = vsb;
      *(ckt->CKTstate0 + inst->NUMOSvgb) = vgb;
      *(ckt->CKTstate0 + inst->NUMOSid) = id;
      *(ckt->CKTstate0 + inst->NUMOSis) = is;
      *(ckt->CKTstate0 + inst->NUMOSig) = ig;
      *(ckt->CKTstate0 + inst->NUMOSdIdDVdb) = g.dIdDVdb;
      *(ckt->CKTstate0 + inst->NUMOSdIdDVsb) = g.dIdDVsb;
      *(ckt->CKTstate0 + inst->NUMOSdIdDVgb) = g.dIdDVgb;
      *(ckt->CKTstate0 + inst->NUMOSdIsDVdb) = g.dIsDVdb;
      *(ckt->CKTstate0 + inst->NUMOSdIsDVsb) = g.dIsDVsb;
      *(ckt->CKTstate0 + inst->NUMOSdIsDVgb) = g.dIsDVgb;
      *(ckt->CKTstate0 + inst->NUMOSdIgDVdb) = g.dIgDVdb;
      *(ckt->CKTstate0 + inst->NUMOSdIgDVsb) = g.dIgDVsb;
      *(ckt->CKTstate0 + inst->NUMOSdIgDVgb) = g.dIgDVgb;

  load:
      /*
       * load current excitation vector
       */

      ideq = id - g.dIdDVdb * vdb - g.dIdDVsb * vsb - g.dIdDVgb * vgb;
      iseq = is - g.dIsDVdb * vdb - g.dIsDVsb * vsb - g.dIsDVgb * vgb;
      igeq = ig - g.dIgDVdb * vdb - g.dIgDVsb * vsb - g.dIgDVgb * vgb;
      *(ckt->CKTrhs + inst->NUMOSdrainNode) -= ideq;
      *(ckt->CKTrhs + inst->NUMOSsourceNode) -= iseq;
      *(ckt->CKTrhs + inst->NUMOSgateNode) -= igeq;
      *(ckt->CKTrhs + inst->NUMOSbulkNode) += ideq + iseq + igeq;
      /*
       * load y matrix
       */

      *(inst->NUMOSdrainDrainPtr) += g.dIdDVdb;
      *(inst->NUMOSdrainSourcePtr) += g.dIdDVsb;
      *(inst->NUMOSdrainGatePtr) += g.dIdDVgb;
      *(inst->NUMOSdrainBulkPtr) -= g.dIdDVdb + g.dIdDVsb + g.dIdDVgb;

      *(inst->NUMOSsourceDrainPtr) += g.dIsDVdb;
      *(inst->NUMOSsourceSourcePtr) += g.dIsDVsb;
      *(inst->NUMOSsourceGatePtr) += g.dIsDVgb;
      *(inst->NUMOSsourceBulkPtr) -= g.dIsDVdb + g.dIsDVsb + g.dIsDVgb;

      *(inst->NUMOSgateDrainPtr) += g.dIgDVdb;
      *(inst->NUMOSgateSourcePtr) += g.dIgDVsb;
      *(inst->NUMOSgateGatePtr) += g.dIgDVgb;
      *(inst->NUMOSgateBulkPtr) -= g.dIgDVdb + g.dIgDVsb + g.dIgDVgb;

      *(inst->NUMOSbulkDrainPtr) -= g.dIdDVdb + g.dIsDVdb + g.dIgDVdb;
      *(inst->NUMOSbulkSourcePtr) -= g.dIdDVsb + g.dIsDVsb + g.dIgDVsb;
      *(inst->NUMOSbulkGatePtr) -= g.dIdDVgb + g.dIsDVgb + g.dIgDVgb;
      *(inst->NUMOSbulkBulkPtr) += g.dIdDVdb + g.dIdDVsb + g.dIdDVgb
	  + g.dIsDVdb + g.dIsDVsb + g.dIsDVgb
	  + g.dIgDVdb + g.dIgDVsb + g.dIgDVgb;

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
NUMOSinitSmSig(NUMOSinstance *inst)
{
  struct mosAdmittances yAc;
  double omega = NUMOSmodPtr(inst)->NUMOSmethods->METHomega;

  AcAnalysisMethod = SOR_ONLY;
  (void) NUMOSadmittance(inst->NUMOSpDevice, omega, &yAc);
  inst->NUMOSc11 = yAc.yIdVdb.imag / omega;
  inst->NUMOSc12 = yAc.yIdVgb.imag / omega;
  inst->NUMOSc13 = yAc.yIdVsb.imag / omega;
  inst->NUMOSc21 = yAc.yIgVdb.imag / omega;
  inst->NUMOSc22 = yAc.yIgVgb.imag / omega;
  inst->NUMOSc23 = yAc.yIgVsb.imag / omega;
  inst->NUMOSc31 = yAc.yIsVdb.imag / omega;
  inst->NUMOSc32 = yAc.yIsVgb.imag / omega;
  inst->NUMOSc33 = yAc.yIsVsb.imag / omega;
  inst->NUMOSy11r = yAc.yIdVdb.real;
  inst->NUMOSy11i = yAc.yIdVdb.imag;
  inst->NUMOSy12r = yAc.yIdVgb.real;
  inst->NUMOSy12i = yAc.yIdVgb.imag;
  inst->NUMOSy13r = yAc.yIdVsb.real;
  inst->NUMOSy13i = yAc.yIdVsb.imag;
  inst->NUMOSy21r = yAc.yIgVdb.real;
  inst->NUMOSy21i = yAc.yIgVdb.imag;
  inst->NUMOSy22r = yAc.yIgVgb.real;
  inst->NUMOSy22i = yAc.yIgVgb.imag;
  inst->NUMOSy23r = yAc.yIgVsb.real;
  inst->NUMOSy23i = yAc.yIgVsb.imag;
  inst->NUMOSy31r = yAc.yIsVdb.real;
  inst->NUMOSy31i = yAc.yIsVdb.imag;
  inst->NUMOSy32r = yAc.yIsVgb.real;
  inst->NUMOSy32i = yAc.yIsVgb.imag;
  inst->NUMOSy33r = yAc.yIsVsb.real;
  inst->NUMOSy33i = yAc.yIsVsb.imag;
  inst->NUMOSsmSigAvail = TRUE;
  return (OK);
}
