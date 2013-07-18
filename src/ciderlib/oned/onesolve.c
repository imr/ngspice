/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Functions needed to calculate solutions for 1D devices.
 */

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onedev.h"
#include "ngspice/onemesh.h"
#include "ngspice/spmatrix.h"
#include "ngspice/bool.h"
#include "ngspice/macros.h"
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/cidersupt.h"
#include "../../maths/misc/norm.h"

#include "ngspice/ifsim.h"
extern IFfrontEnd *SPfrontEnd;



/* The iteration driving loop and convergence check */
void
ONEdcSolve(ONEdevice *pDevice, int iterationLimit, BOOLEAN newSolver, 
           BOOLEAN tranAnalysis, ONEtranInfo *info)
{
  ONEnode *pNode;
  ONEelem *pElem;
  int index, eIndex, error;
  int timesConverged = 0, negConc = FALSE;
  int size = pDevice->numEqns;
  BOOLEAN quitLoop;
  BOOLEAN debug = FALSE;
  double *rhs = pDevice->rhs;
/*  double *intermediate = pDevice->copiedSolution; */
  double *solution = pDevice->dcSolution;
  double *delta = pDevice->dcDeltaSolution;
  double poissNorm, contNorm;
  double startTime, totalStartTime;
  double totalTime, loadTime, factorTime, solveTime, updateTime, checkTime;
  double orderTime = 0.0;

  quitLoop = FALSE;
  debug =   (!tranAnalysis && ONEdcDebug) 
          ||(tranAnalysis && ONEtranDebug); 
  pDevice->iterationNumber = 0;
  pDevice->converged = FALSE;


  totalTime = loadTime = factorTime = solveTime = updateTime = checkTime = 0.0;
  totalStartTime = SPfrontEnd->IFseconds();

  if (debug) {
    if (pDevice->poissonOnly) {
      fprintf(stdout, "Equilibrium Solution:\n");
    } else {
      fprintf(stdout, "Bias Solution:\n");
    }
    fprintf(stdout, "Iteration  RHS Norm\n");
  }
  while (! (pDevice->converged 
         || pDevice->iterationNumber > iterationLimit
	 || quitLoop)) {
    pDevice->iterationNumber++;

    if ((!pDevice->poissonOnly) && (iterationLimit > 0)
	&&(!tranAnalysis)) {
      ONEjacCheck(pDevice, tranAnalysis, info);
    }
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    if (pDevice->poissonOnly) {
      ONEQsysLoad(pDevice);
    } else {
      ONE_sysLoad(pDevice, tranAnalysis, info);
    }
    pDevice->rhsNorm = maxNorm(rhs, size);
    loadTime += SPfrontEnd->IFseconds() - startTime;
    if (debug) {
      fprintf(stdout, "%7d   %11.4e%s\n",
	  pDevice->iterationNumber - 1, pDevice->rhsNorm,
	  negConc ? "   negative conc encountered" : "");
      negConc = FALSE;
    }
    /* FACTOR */
    startTime = SPfrontEnd->IFseconds();
    error = spFactor(pDevice->matrix);
    factorTime += SPfrontEnd->IFseconds() - startTime;
    if (newSolver) {
      if (pDevice->iterationNumber == 1) {
	orderTime = factorTime;
      } else if (pDevice->iterationNumber == 2) {
	orderTime -= factorTime - orderTime;
	factorTime -= orderTime;
	if (pDevice->poissonOnly) {
	  pDevice->pStats->orderTime[STAT_SETUP] += orderTime;
	} else {
	  pDevice->pStats->orderTime[STAT_DC] += orderTime;
	}
	newSolver = FALSE;
      }
    }
    if (foundError(error)) {
      if (error == spSINGULAR) {
	int badRow, badCol;
	spWhereSingular(pDevice->matrix, &badRow, &badCol);
	printf("*****  singular at (%d,%d)\n", badRow, badCol);
      }
      /* Should probably try to recover now, but we'll punt instead. */
      exit(-1);
    }
    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhs, delta, NULL, NULL);
    solveTime += SPfrontEnd->IFseconds() - startTime;

    /* UPDATE */
    startTime = SPfrontEnd->IFseconds();
    /*
     * Use norm reducing Newton method only for DC bias solutions. Since norm
     * reducing can get trapped by numerical errors, turn it off when we are
     * somewhat close to the solution.
     */
    if ((!pDevice->poissonOnly) && (iterationLimit > 0)
	&& (!tranAnalysis) && (pDevice->rhsNorm > 1e-6)) {
      error = ONEnewDelta(pDevice, tranAnalysis, info);
      if (error) {
	pDevice->converged = FALSE;
	quitLoop = TRUE;
	updateTime += SPfrontEnd->IFseconds() - startTime;
	continue;
      }
    }
    for (index = 1; index <= size; index++) {
      solution[index] += delta[index];
    }
    updateTime += SPfrontEnd->IFseconds() - startTime;

    /* CHECK CONVERGENCE */
    startTime = SPfrontEnd->IFseconds();
    /* Check if updates have gotten sufficiently small. */
    if (pDevice->iterationNumber != 1) {
      /*
       * pDevice->converged = ONEdeltaConverged(pDevice);
       */
      pDevice->converged = ONEpsiDeltaConverged(pDevice, &negConc);
    }
    /* Check if the rhs residual is smaller than abstol. */
    if (pDevice->converged &&(!pDevice->poissonOnly)
	&&(!tranAnalysis)) {
      ONE_rhsLoad(pDevice, tranAnalysis, info);
      pDevice->rhsNorm = maxNorm(rhs, size);
      if (pDevice->rhsNorm > pDevice->abstol) {
	pDevice->converged = FALSE;
      }
      if ((++timesConverged >= 2)
	  &&(pDevice->rhsNorm < 1e3 * pDevice->abstol)) {
	pDevice->converged = TRUE;
      } else if (timesConverged >= 5) {
	pDevice->converged = FALSE;
	quitLoop = TRUE;
      }
    } else if (pDevice->converged && pDevice->poissonOnly) {
      ONEQrhsLoad(pDevice);
      pDevice->rhsNorm = maxNorm(rhs, size);
      if (pDevice->rhsNorm > pDevice->abstol) {
	pDevice->converged = FALSE;
      }
      if (++timesConverged >= 5) {
	pDevice->converged = TRUE;
      }
    }
    /* Check if any of the carrier concentrations are negative. */
    if (pDevice->converged &&(!pDevice->poissonOnly)) {
      /* Clear garbage entry since carrier-free elements reference it */
      solution[0] = 0.0;

      for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
	pElem = pDevice->elemArray[eIndex];
	for (index = 0; index <= 1; index++) {
	  if (pElem->evalNodes[index]) {
	    pNode = pElem->pNodes[index];
	    if (solution[pNode->nEqn] < 0.0) {
	      pDevice->converged = FALSE;
	      negConc = TRUE;
	      if (tranAnalysis) {
		quitLoop = TRUE;
	      } else {
		solution[pNode->nEqn] = 0.0;
	      }
	    }
	    if (solution[pNode->pEqn] < 0.0) {
	      pDevice->converged = FALSE;
	      negConc = TRUE;
	      if (tranAnalysis) {
		quitLoop = TRUE;
	      } else {
		solution[pNode->pEqn] = 0.0;
	      }
	    }
	  }
	}
      }
      /* Set to a consistent state if negative conc was encountered. */
      if (!pDevice->converged) {
	ONE_rhsLoad(pDevice, tranAnalysis, info);
	pDevice->rhsNorm = maxNorm(rhs, size);
      }
    }
    checkTime += SPfrontEnd->IFseconds() - startTime;
  }
  totalTime += SPfrontEnd->IFseconds() - totalStartTime;

  if (tranAnalysis) {
    pDevice->pStats->loadTime[STAT_TRAN] += loadTime;
    pDevice->pStats->factorTime[STAT_TRAN] += factorTime;
    pDevice->pStats->solveTime[STAT_TRAN] += solveTime;
    pDevice->pStats->updateTime[STAT_TRAN] += updateTime;
    pDevice->pStats->checkTime[STAT_TRAN] += checkTime;
    pDevice->pStats->numIters[STAT_TRAN] += pDevice->iterationNumber;
  } else if (pDevice->poissonOnly) {
    pDevice->pStats->loadTime[STAT_SETUP] += loadTime;
    pDevice->pStats->factorTime[STAT_SETUP] += factorTime;
    pDevice->pStats->solveTime[STAT_SETUP] += solveTime;
    pDevice->pStats->updateTime[STAT_SETUP] += updateTime;
    pDevice->pStats->checkTime[STAT_SETUP] += checkTime;
    pDevice->pStats->numIters[STAT_SETUP] += pDevice->iterationNumber;
  } else {
    pDevice->pStats->loadTime[STAT_DC] += loadTime;
    pDevice->pStats->factorTime[STAT_DC] += factorTime;
    pDevice->pStats->solveTime[STAT_DC] += solveTime;
    pDevice->pStats->updateTime[STAT_DC] += updateTime;
    pDevice->pStats->checkTime[STAT_DC] += checkTime;
    pDevice->pStats->numIters[STAT_DC] += pDevice->iterationNumber;
  }

  if (debug) {
    if (!tranAnalysis) {
      pDevice->rhsNorm = maxNorm(rhs, size);
      fprintf(stdout, "%7d   %11.4e%s\n",
	pDevice->iterationNumber, pDevice->rhsNorm,
	negConc ? "   negative conc in solution" : "");
    }
    if (pDevice->converged) {
      if (!pDevice->poissonOnly) {
	rhs[0] = 0.0;
	poissNorm = contNorm = 0.0;
	for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
	  pElem = pDevice->elemArray[eIndex];
	  for (index = 0; index <= 1; index++) {
	    if (pElem->evalNodes[index]) {
	      pNode = pElem->pNodes[index];
	      poissNorm = MAX(poissNorm, ABS(rhs[pNode->psiEqn]));
	      contNorm = MAX(contNorm, ABS(rhs[pNode->nEqn]));
	      contNorm = MAX(contNorm, ABS(rhs[pNode->pEqn]));
	    }
	  }
	}
	fprintf(stdout,
	    "Residual: %11.4e C/um^2 poisson, %11.4e A/um^2 continuity\n",
	    poissNorm * EpsNorm * ENorm * 1e-8,
	    contNorm * JNorm * 1e-8);
      } else {
	fprintf(stdout, "Residual: %11.4e C/um^2 poisson\n",
	    pDevice->rhsNorm * EpsNorm * ENorm * 1e-8);
      }
    }
  }
}

/*
 * A function that checks convergence based on the convergence of the quasi
 * Fermi levels. In theory, this should work better than the one currently
 * being used since we are always looking at potentials: (psi, phin, phip).
 */
BOOLEAN
ONEpsiDeltaConverged(ONEdevice *pDevice, int *pNegConc)
{
  int index, nIndex, eIndex;
  ONEnode *pNode;
  ONEelem *pElem;
  double xOld, xNew, xDelta, tol;
  double psi, newPsi, nConc, pConc, newN, newP;
  double phiN, phiP, newPhiN, newPhiP;
  BOOLEAN converged = TRUE;

  /* Equilibrium solution. */
  if (pDevice->poissonOnly) {
    for (index = 1; index <= pDevice->numEqns; index++) {
      xOld = pDevice->dcSolution[index];
      xDelta = pDevice->dcDeltaSolution[index];
      xNew = xOld + xDelta;
      tol = pDevice->abstol + pDevice->reltol * MAX(ABS(xOld), ABS(xNew));
      if (ABS(xDelta) > tol) {
	converged = FALSE;
	goto done;
      }
    }
    return (converged); 

  }
  /* Bias solution. Check convergence on psi, phin, phip. */
  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pNode->nodeType != CONTACT) {
	  /* check convergence on psi */
	  xOld = pDevice->dcSolution[pNode->psiEqn];
	  xDelta = pDevice->dcDeltaSolution[pNode->psiEqn];
	  xNew = xOld + xDelta;
	  tol = pDevice->abstol +
	      pDevice->reltol * MAX(ABS(xOld), ABS(xNew));
	  if (ABS(xDelta) > tol) {
	    converged = FALSE;
	    goto done;
	  }
	}
	/* check convergence on phin and phip */
	if (pElem->elemType == SEMICON && pNode->nodeType != CONTACT) {
	  psi = pDevice->dcSolution[pNode->psiEqn];
	  nConc = pDevice->dcSolution[pNode->nEqn];
	  pConc = pDevice->dcSolution[pNode->pEqn];
	  newPsi = psi + pDevice->dcDeltaSolution[pNode->psiEqn];
	  newN = nConc + pDevice->dcDeltaSolution[pNode->nEqn];
	  newP = pConc + pDevice->dcDeltaSolution[pNode->pEqn];
	  if (newN <= 0.0 || newP <= 0.0) {
	    *pNegConc = TRUE;
	    converged = FALSE;
	    goto done;
	  }
	  phiN = psi - log(nConc / pNode->nie);
	  phiP = psi + log(pConc / pNode->nie);
	  newPhiN = newPsi - log(newN / pNode->nie);
	  newPhiP = newPsi + log(newP / pNode->nie);
	  tol = pDevice->abstol + pDevice->reltol * MAX(ABS(phiN), ABS(newPhiN));
	  if (ABS(newPhiN - phiN) > tol) {
	    converged = FALSE;
	    goto done;
	  }
	  tol = pDevice->abstol + pDevice->reltol * MAX(ABS(phiP), ABS(newPhiP));
	  if (ABS(newPhiP - phiP) > tol) {
	    converged = FALSE;
	    goto done;
	  }
	}
      }
    }
  }
done:

  return (converged);
}

/*
 * See if the update to the solution is small enough. Returns TRUE if it is.
 */
BOOLEAN
ONEdeltaConverged(ONEdevice *pDevice)
{
  int index;
  BOOLEAN converged = TRUE;
  double *solution = pDevice->dcSolution;
  double *delta = pDevice->dcDeltaSolution;
  double xOld, xNew, tol;


  for (index = 1; index <= pDevice->numEqns; index++) {
    xOld = solution[index];
    xNew = xOld + delta[index];
    tol = pDevice->abstol + pDevice->reltol * MAX(ABS(xOld), ABS(xNew));
    if (ABS(xOld - xNew) > tol) {
      converged = FALSE;
      break;
    }
  }
  return (converged);
}

/*
 * See if the update to the solution is small enough and the solution is
 * physically reasonable. Returns TRUE if it is.
 */
BOOLEAN
ONEdeviceConverged(ONEdevice *pDevice)
{
  int index, eIndex;
  BOOLEAN converged = TRUE;
  double *solution = pDevice->dcSolution;
  ONEnode *pNode;
  ONEelem *pElem;
  double startTime;

  /*
   * If the update is sufficently small, and the carrier concentrations are
   * all positive, then return TRUE, else return FALSE.
   */
   
  
  /* CHECK CONVERGENCE */
  startTime = SPfrontEnd->IFseconds();
  if ((converged = ONEdeltaConverged(pDevice)) == TRUE) {
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nEqn != 0 && solution[pNode->nEqn] < 0.0) {
	    converged = FALSE;
	    solution[pNode->nEqn] = pNode->nConc = 0.0;
	  }
	  if (pNode->pEqn != 0 && solution[pNode->pEqn] < 0.0) {
	    converged = FALSE;
	    solution[pNode->pEqn] = pNode->pConc = 0.0;
	  }
	}
      }
    }
  }
  pDevice->pStats->checkTime[STAT_TRAN] += SPfrontEnd->IFseconds() - startTime;

  return (converged); 
}

/*
 * Load and factor the Jacobian so that it is consistent with the current
 * solution.
 */
void
ONEresetJacobian(ONEdevice *pDevice)
{
  int error;
  
  
  ONE_jacLoad(pDevice);
  error = spFactor(pDevice->matrix);
  if (foundError(error)) {
    exit(-1);
  }
}

/*
 * Compute the device state assuming charge neutrality exists everywhere in
 * the device.  In insulators, where there is no charge, assign the potential
 * at half the insulator band gap (refPsi).
 */
void
ONEstoreNeutralGuess(ONEdevice *pDevice)
{
  int nIndex, eIndex;
  ONEelem *pElem;
  ONEnode *pNode;
  double nie, conc, absConc, refPsi, psi, ni, pi, sign;


  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    refPsi = pElem->matlInfo->refPsi;
    if (pElem->elemType == INSULATOR) {
      for (nIndex = 0; nIndex <= 1; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  if (pNode->nodeType == CONTACT) {
	    /*
	     * A metal contact to insulating domain, so use work function
	     * difference as the value of psi.
	     */
	    pNode->psi = RefPsi - pNode->eaff;
	  } else {
	    pNode->psi = refPsi;
	  }
	}
      }
    }
    if (pElem->elemType == SEMICON) {
      for (nIndex = 0; nIndex <= 1; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  nie = pNode->nie;
	  conc = pNode->netConc / nie;
	  psi = 0.0;
	  ni = nie;
	  pi = nie;
	  sign = SGN(conc);
	  absConc = ABS(conc);
	  if (conc != 0.0) {
	    psi = sign * log(0.5 * absConc + sqrt(1.0 + 0.25 * absConc * absConc));
	    ni = nie * exp(psi);
	    pi = nie * exp(-psi);
	    if (FreezeOut) {
	      /* Use Newton's Method to solve for psi. */
	      int ctr, maxiter = 10;
	      double rhs, deriv, fNa, fNd, fdNa, fdNd;
	      for (ctr = 0; ctr < maxiter; ctr++) {
		pNode->nConc = ni;
		pNode->pConc = pi;
		ONEQfreezeOut(pNode, &fNd, &fNa, &fdNd, &fdNa);
		rhs = pi - ni + pNode->nd * fNd - pNode->na * fNa;
		deriv = pi + ni - pNode->nd * fdNd + pNode->na * fdNa;
		psi += rhs / deriv;
		ni = nie * exp(psi);
		pi = nie * exp(-psi);
	      }
	    }
	  }
	  pNode->psi = refPsi + psi;
	  pNode->psi0 = pNode->psi;
	  pNode->vbe = refPsi;
	  pNode->nConc = ni;
	  pNode->pConc = pi;
	  /* Now store the initial guess in the dc solution vector. */
	  pDevice->dcSolution[pNode->poiEqn] = pNode->psi;
	}
      }
    }
  }
}

/*
 * Compute the device state at thermal equilibrium. This state is equal to
 * the solution of just Poisson's equation. The charge-neutral solution is
 * taken as an initial guess.
 */
void
ONEequilSolve(ONEdevice *pDevice)
{
  BOOLEAN newSolver = FALSE;
  int error;
  int nIndex, eIndex;
  ONEelem *pElem;
  ONEnode *pNode;
  double startTime, setupTime, miscTime;


  setupTime = miscTime = 0.0;

  /* SETUP */
  startTime = SPfrontEnd->IFseconds();
  switch (pDevice->solverType) {
  case SLV_SMSIG:
  case SLV_BIAS:
    /* free up memory allocated for the bias solution */
    FREE(pDevice->dcSolution);
    FREE(pDevice->dcDeltaSolution);
    FREE(pDevice->copiedSolution);
    FREE(pDevice->rhs);
    FREE(pDevice->rhsImag);
    spDestroy(pDevice->matrix);
    /* FALLTHRU */
  case SLV_NONE:
    pDevice->poissonOnly = TRUE;
    pDevice->numEqns = pDevice->dimEquil - 1;
    XCALLOC(pDevice->dcSolution, double, pDevice->dimEquil);
    XCALLOC(pDevice->dcDeltaSolution, double, pDevice->dimEquil);
    XCALLOC(pDevice->copiedSolution, double, pDevice->dimEquil);
    XCALLOC(pDevice->rhs, double, pDevice->dimEquil);
    pDevice->matrix = spCreate(pDevice->numEqns, 0, &error);
    if (error == spNO_MEMORY) {
      printf("ONEequilSolve: Out of Memory\n");
      exit(-1);
    }
    newSolver = TRUE;
    spSetReal(pDevice->matrix);
    ONEQjacBuild(pDevice);
    pDevice->numOrigEquil = spElementCount(pDevice->matrix);
    pDevice->numFillEquil = 0;
    /* FALLTHRU */
  case SLV_EQUIL:
    pDevice->solverType = SLV_EQUIL;
    break;
  default:
    fprintf(stderr, "Panic: Unknown solver type in equil solution.\n");
    exit(-1);
    break;
  }
  ONEstoreNeutralGuess(pDevice);
  setupTime += SPfrontEnd->IFseconds() - startTime;

  /* SOLVE */
  ONEdcSolve(pDevice, MaxIterations, newSolver, FALSE, NULL);

  /* MISCELLANEOUS */
  startTime = SPfrontEnd->IFseconds();
  if (newSolver) {
    pDevice->numFillEquil = spFillinCount(pDevice->matrix);
  }
  if (pDevice->converged) {
    ONEQcommonTerms(pDevice);

    /* Save equilibrium potential. */
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (nIndex = 0; nIndex <= 1; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  pNode->psi0 = pNode->psi;
	}
      }
    }
  } else {
    printf("ONEequilSolve: No Convergence\n");
  }
  miscTime += SPfrontEnd->IFseconds() - startTime;
  pDevice->pStats->setupTime[STAT_SETUP] += setupTime;
  pDevice->pStats->miscTime[STAT_SETUP] += miscTime;
}

/*
 * Compute the device state under an applied bias. The equilibrium solution
 * is taken as an initial guess the first time this is called.
 */
void
ONEbiasSolve(ONEdevice *pDevice, int iterationLimit, 
             BOOLEAN tranAnalysis, ONEtranInfo *info)
{
  BOOLEAN newSolver = FALSE;
  int error;
  int index, eIndex;
  double *solution;
  ONEelem *pElem;
  ONEnode *pNode;
  double startTime, setupTime, miscTime;


  setupTime = miscTime = 0.0;


  /* SETUP */
  startTime = SPfrontEnd->IFseconds();
  switch (pDevice->solverType) {
  case SLV_EQUIL:
    /* Free up the vectors allocated in the equilibrium solution. */
    FREE(pDevice->dcSolution);
    FREE(pDevice->dcDeltaSolution);
    FREE(pDevice->copiedSolution);
    FREE(pDevice->rhs);
    spDestroy(pDevice->matrix);
    /* FALLTHRU */
  case SLV_NONE:
    pDevice->poissonOnly = FALSE;
    pDevice->numEqns = pDevice->dimBias - 1;
    XCALLOC(pDevice->dcSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->dcDeltaSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->copiedSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->rhs, double, pDevice->dimBias);
    XCALLOC(pDevice->rhsImag, double, pDevice->dimBias);
    pDevice->matrix = spCreate(pDevice->numEqns, 1, &error);
    if (error == spNO_MEMORY) {
      exit(-1);
    }
    newSolver = TRUE;
    ONE_jacBuild(pDevice);
    pDevice->numOrigBias = spElementCount(pDevice->matrix);
    pDevice->numFillBias = 0;
    ONEstoreInitialGuess(pDevice);
    /* FALLTHRU */
  case SLV_SMSIG:
    spSetReal(pDevice->matrix);
    /* FALLTHRU */
  case SLV_BIAS:
    pDevice->solverType = SLV_BIAS;
    break;
  default:
    fprintf(stderr, "Panic: Unknown solver type in bias solution.\n");
    exit(-1);
    break;
  }
  setupTime += SPfrontEnd->IFseconds() - startTime;

  /* SOLVE */
  ONEdcSolve(pDevice, iterationLimit, newSolver, tranAnalysis, info);

  /* MISCELLANEOUS */
  startTime = SPfrontEnd->IFseconds();
  if (newSolver) {
    pDevice->numFillBias = spFillinCount(pDevice->matrix);
  }
  solution = pDevice->dcSolution;
  if ((!pDevice->converged) && iterationLimit > 1) {
  } else if (pDevice->converged) {
    /* Update the nodal quantities. */
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->psiEqn != 0) {
	    pNode->psi = solution[pNode->psiEqn];
	  }
	  if (pNode->nEqn != 0) {
	    pNode->nConc = solution[pNode->nEqn];
	  }
	  if (pNode->pEqn != 0) {
	    pNode->pConc = solution[pNode->pEqn];
	  }
	}
      }
    }
    /* Update the current terms. */
    ONE_commonTerms(pDevice, FALSE, tranAnalysis, info);
  } else if (iterationLimit <= 1) {
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    pNode->psi = solution[pNode->psiEqn];
	    pDevice->devState0 [pNode->nodePsi] = pNode->psi;
	    if (pElem->elemType == SEMICON) {
	      pNode->nConc = solution[pNode->nEqn];
	      pNode->pConc = solution[pNode->pEqn];
	      pDevice->devState0 [pNode->nodeN] = pNode->nConc;
	      pDevice->devState0 [pNode->nodeP] = pNode->pConc;
	    }
	  }
	}
      }
    }
  }
  miscTime += SPfrontEnd->IFseconds() - startTime;
  if (tranAnalysis) {
    pDevice->pStats->setupTime[STAT_TRAN] += setupTime;
    pDevice->pStats->miscTime[STAT_TRAN] += miscTime;
  } else {
    pDevice->pStats->setupTime[STAT_DC] += setupTime;
    pDevice->pStats->miscTime[STAT_DC] += miscTime;
  }
}

/* Copy the device's equilibrium state into the solution vector. */
void
ONEstoreEquilibGuess(ONEdevice *pDevice)
{
  int nIndex, eIndex;
  double *solution = pDevice->dcSolution;
  double refPsi;
  ONEelem *pElem;
  ONEnode *pNode;


  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    refPsi = pElem->matlInfo->refPsi;
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pNode->nodeType != CONTACT) {
	  solution[pNode->psiEqn] = pNode->psi0;
	  if (pElem->elemType == SEMICON) {
	    solution[pNode->nEqn] = pNode->nie * exp(pNode->psi0 - refPsi);
	    solution[pNode->pEqn] = pNode->nie * exp(-pNode->psi0 + refPsi);
	  }
	}
      }
    }
  }
}

/* Copy the device's internal state into the solution vector. */
void
ONEstoreInitialGuess(ONEdevice *pDevice)
{
  int nIndex, eIndex;
  double *solution = pDevice->dcSolution;
  ONEelem *pElem;
  ONEnode *pNode;


  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pNode->nodeType != CONTACT) {
	  solution[pNode->psiEqn] = pNode->psi;
	  if (pElem->elemType == SEMICON) {
	    solution[pNode->nEqn] = pNode->nConc;
	    solution[pNode->pEqn] = pNode->pConc;
	  }
	}
      }
    }
  }
}


int
ONEnewDelta(ONEdevice *pDevice, BOOLEAN tranAnalysis, ONEtranInfo *info)
{
  int index, iterNum;
  double newNorm, fib, lambda, fibn, fibp;
  BOOLEAN acceptable = FALSE, error = FALSE;
  iterNum = 0;
  lambda = 1.0;
  fibn = 1.0;
  fibp = 1.0;
  
  
  /*
   * Copy the contents of dcSolution into copiedSolution and modify
   * dcSolution by adding the deltaSolution.
   */
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->copiedSolution[index] = pDevice->dcSolution[index];
    pDevice->dcSolution[index] += pDevice->dcDeltaSolution[index];
  }

  if (pDevice->poissonOnly) {
    ONEQrhsLoad(pDevice);
  } else {
    ONE_rhsLoad(pDevice, tranAnalysis, info);
  }
  newNorm = maxNorm(pDevice->rhs, pDevice->numEqns);
  if (pDevice->rhsNorm <= pDevice->abstol) {
    lambda = 0.0;
    newNorm = pDevice->rhsNorm;
  } else if (newNorm < pDevice->rhsNorm) {
    acceptable = TRUE;
  } else {
    /* chop the step size so that deltax is acceptable */
    if (ONEdcDebug) {
      fprintf(stdout, "          %11.4e  %11.4e\n",
	  newNorm, lambda);
    }
    while (!acceptable) {
      iterNum++;

      if (iterNum > NORM_RED_MAXITERS) {
	error = TRUE;
	lambda = 0.0;
	/* Don't break out until after we've reset the device. */
      }
      fib = fibp;
      fibp = fibn;
      fibn += fib;
      lambda *= (fibp / fibn);

      for (index = 1; index <= pDevice->numEqns; index++) {
	pDevice->dcSolution[index] = pDevice->copiedSolution[index]
	    + lambda * pDevice->dcDeltaSolution[index];
      }
      if (pDevice->poissonOnly) {
	ONEQrhsLoad(pDevice);
      } else {
	ONE_rhsLoad(pDevice, tranAnalysis, info);
      }
      newNorm = maxNorm(pDevice->rhs, pDevice->numEqns);
      if (error) {
	break;
      }
      if (newNorm <= pDevice->rhsNorm) {
	acceptable = TRUE;
      }
      if (ONEdcDebug) {
	fprintf(stdout, "          %11.4e  %11.4e\n",
	    newNorm, lambda);
      }
    }
  }
  /* Restore the previous dcSolution and store the new deltaSolution. */
  pDevice->rhsNorm = newNorm;
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->dcSolution[index] = pDevice->copiedSolution[index];
    pDevice->dcDeltaSolution[index] *= lambda;
  }
  return(error);
}


/* Predict the values of the internal variables at the next timepoint. */
/* Needed for Predictor-Corrector LTE estimation */
void
ONEpredict(ONEdevice *pDevice, ONEtranInfo *info)
{
  int nIndex, eIndex;
  ONEnode *pNode;
  ONEelem *pElem;
  double startTime, miscTime = 0.0;


  /* TRANSIENT MISC */
  startTime = SPfrontEnd->IFseconds();
  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	pNode->psi = pDevice->devState1 [pNode->nodePsi];
	if (pElem->elemType == SEMICON && pNode->nodeType != CONTACT) {
	  pNode->nPred = predict(pDevice->devStates, info, pNode->nodeN);
	  pNode->pPred = predict(pDevice->devStates, info, pNode->nodeP);
	  pNode->nConc = pNode->nPred;
	  pNode->pConc = pNode->pPred;
	}
      }
    }
  }
  miscTime += SPfrontEnd->IFseconds() - startTime;
  pDevice->pStats->miscTime[STAT_TRAN] += miscTime;
}


/* Estimate the device's overall truncation error. */
double
ONEtrunc(ONEdevice *pDevice, ONEtranInfo *info, double delta)
{
  int nIndex, eIndex;
  ONEelem *pElem;
  ONEnode *pNode;
  double tolN, tolP, lte, relError, temp;
  double lteCoeff = info->lteCoeff;
  double mult = 10.0;
  double reltol;
  double startTime, lteTime = 0.0;
   
 
  /* TRANSIENT LTE */
  startTime = SPfrontEnd->IFseconds();

  relError = 0.0;
  reltol = pDevice->reltol * mult;

  /* Need to get the predictor for the current order. */
  /* The scheme currently used is very dumb. Need to fix later. */
  /* XXX Why is the scheme dumb?  Never understood this. */
  computePredCoeff(info->method, info->order, info->predCoeff, info->delta);

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pElem->elemType == SEMICON && pNode->nodeType != CONTACT) {
	  tolN = pDevice->abstol + reltol * ABS(pNode->nConc);
	  tolP = pDevice->abstol + reltol * ABS(pNode->pConc);
	  pNode->nPred = predict(pDevice->devStates, info, pNode->nodeN);
	  pNode->pPred = predict(pDevice->devStates, info, pNode->nodeP);
	  lte = lteCoeff * (pNode->nConc - pNode->nPred);
	  temp = lte / tolN;
	  relError += temp * temp;
	  lte = lteCoeff * (pNode->pConc - pNode->pPred);
	  temp = lte / tolP;
	  relError += temp * temp;
	}
      }
    }
  }

  /* Make sure error is non-zero. */
  relError = MAX(pDevice->abstol, relError);

  /* The total relative error has been calculated norm-2 squared. */
  relError = sqrt(relError / pDevice->numEqns);

  /* Use the order of the integration method to compute new delta. */
  temp = delta / pow(relError, 1.0 / (info->order + 1));

  lteTime += SPfrontEnd->IFseconds() - startTime;
  pDevice->pStats->lteTime += lteTime;

  return (temp);
}

/* Save info from state table into the internal state. */
void
ONEsaveState(ONEdevice *pDevice)
{
  int nIndex, eIndex;
  ONEnode *pNode;
  ONEelem *pElem;

  for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
    pElem = pDevice->elemArray[eIndex];
    for (nIndex = 0; nIndex <= 1; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	pNode->psi = pDevice->devState1 [pNode->nodePsi];
	if (pElem->elemType == SEMICON && pNode->nodeType != CONTACT) {
	  pNode->nConc = pDevice->devState1 [pNode->nodeN];
	  pNode->pConc = pDevice->devState1 [pNode->nodeP];
	}
      }
    }
  }
}

/* Function to compute Nu norm of the rhs vector. */
/* Nu-norm calculation based upon work done at Stanford. */
double
ONEnuNorm(ONEdevice *pDevice)
{
  /* The LU Decomposed matrix is available.  Use it to calculate x. */
  spSolve(pDevice->matrix, pDevice->rhs, pDevice->rhsImag,
      NULL, NULL);

  /* Compute L2-norm of the solution vector (stored in rhsImag) */
  return (l2Norm(pDevice->rhsImag, pDevice->numEqns));
}


/*
 * Check for numerical errors in the Jacobian.  Useful debugging aid when new
 * models are being incorporated.
 */
void
ONEjacCheck(ONEdevice *pDevice, BOOLEAN tranAnalysis, ONEtranInfo *info)
{
  int index, rIndex;
  double del, diff, tol, *dptr;


  if (ONEjacDebug) {
    ONE_sysLoad(pDevice, tranAnalysis, info);
    /*
     * spPrint( pDevice->matrix, 0, 1, 1 );
     */
    pDevice->rhsNorm = maxNorm(pDevice->rhs, pDevice->numEqns);
    for (index = 1; index <= pDevice->numEqns; index++) {
      if (1e3 * ABS(pDevice->rhs[index]) > pDevice->rhsNorm) {
	fprintf(stderr, "eqn %d: res %11.4e, norm %11.4e\n",
	    index, pDevice->rhs[index], pDevice->rhsNorm);
      }
    }
    for (index = 1; index <= pDevice->numEqns; index++) {
      pDevice->rhsImag[index] = pDevice->rhs[index];
    }
    for (index = 1; index <= pDevice->numEqns; index++) {
      pDevice->copiedSolution[index] = pDevice->dcSolution[index];
      del = 1e-4 * pDevice->abstol + 1e-6 * ABS(pDevice->dcSolution[index]);
      pDevice->dcSolution[index] += del;
      ONE_rhsLoad(pDevice, tranAnalysis, info);
      pDevice->dcSolution[index] = pDevice->copiedSolution[index];
      for (rIndex = 1; rIndex <= pDevice->numEqns; rIndex++) {
	diff = (pDevice->rhsImag[rIndex] - pDevice->rhs[rIndex]) / del;
	dptr = spFindElement(pDevice->matrix, rIndex, index);
	/*
	 * if ( dptr ISNOT NULL ) { fprintf( stderr, "[%d][%d]: FD =
	 * %11.4e, AJ = %11.4e\n", rIndex, index, diff, *dptr ); } else {
	 * fprintf( stderr, "[%d][%d]: FD = %11.4e, AJ = %11.4e\n", rIndex,
	 * index, diff, 0.0 ); }
	 */
	if (dptr != NULL) {
	  tol = (1e-4 * pDevice->abstol) + (1e-2 * MAX(ABS(diff), ABS(*dptr)));
	  if ((diff != 0.0) && (ABS(diff - *dptr) > tol)) {
	    fprintf(stderr, "Mismatch[%d][%d]: FD = %11.4e, AJ = %11.4e\n\t FD-AJ = %11.4e vs. %11.4e\n",
		rIndex, index, diff, *dptr,
		ABS(diff - *dptr), tol);
	  }
	} else {
	  if (diff != 0.0) {
	    fprintf(stderr, "Missing [%d][%d]: FD = %11.4e, AJ = 0.0\n",
		rIndex, index, diff);
	  }
	}
      }
    }
  }
}
