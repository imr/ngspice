/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "../../maths/misc/norm.h"
#include "ngspice/bool.h"
#include "ngspice/cidersupt.h"
#include "ngspice/cpextern.h"
#include "ngspice/ifsim.h"
#include "ngspice/macros.h"
#include "ngspice/ngspice.h"
#include "ngspice/numenum.h"
#include "ngspice/numglobs.h"
#include "ngspice/spmatrix.h"
#include "ngspice/twodev.h"
#include "ngspice/twomesh.h"
#include "twoddefs.h"
#include "twodext.h"
extern IFfrontEnd *SPfrontEnd;


/* functions to calculate the 2D solutions */


/* the iteration driving loop and convergence check */
void
TWOdcSolve(TWOdevice *pDevice, int iterationLimit, BOOLEAN newSolver, 
           BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int size = pDevice->numEqns;
  int index, eIndex, error;
  int timesConverged = 0;
  BOOLEAN quitLoop;
  BOOLEAN debug;
  BOOLEAN negConc = FALSE;
  double *rhs = pDevice->rhs;
  double *solution = pDevice->dcSolution;
  double *delta = pDevice->dcDeltaSolution;
  double poissNorm, contNorm;
  double startTime, totalStartTime;
  double totalTime, loadTime, factorTime, solveTime, updateTime, checkTime;
  double orderTime = 0.0;

  totalTime = loadTime = factorTime = solveTime = updateTime = checkTime = 0.0;
  totalStartTime = SPfrontEnd->IFseconds();

  quitLoop = FALSE;
  debug = (!tranAnalysis && TWOdcDebug) || (tranAnalysis && TWOtranDebug);
  pDevice->iterationNumber = 0;
  pDevice->converged = FALSE;

  if (debug) {
    if (pDevice->poissonOnly) {
      fprintf(stdout, "Equilibrium Solution:\n");
    } else {
      fprintf(stdout, "Bias Solution:\n");
    }
    fprintf(stdout, "Iteration  RHS Norm\n");
  }
  while (!(pDevice->converged || (pDevice->iterationNumber > iterationLimit)
	  || quitLoop)) {
    pDevice->iterationNumber++;

    if ((!pDevice->poissonOnly) && (iterationLimit > 0)
	&&(!tranAnalysis)) {
      TWOjacCheck(pDevice, tranAnalysis, info);
    }

    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    if (pDevice->poissonOnly) {
      TWOQsysLoad(pDevice);
    } else if (!OneCarrier) {
      TWO_sysLoad(pDevice, tranAnalysis, info);
    } else if (OneCarrier == N_TYPE) {
      TWONsysLoad(pDevice, tranAnalysis, info);
    } else if (OneCarrier == P_TYPE) {
      TWOPsysLoad(pDevice, tranAnalysis, info);
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
        }
        else if (pDevice->iterationNumber == 2) {
            orderTime -= factorTime - orderTime;
            factorTime -= orderTime;
            if (pDevice->poissonOnly) {
                pDevice->pStats->orderTime[STAT_SETUP] += orderTime;
            }
            else {
                pDevice->pStats->orderTime[STAT_DC] += orderTime;
            }
            /* After first two iterations, no special handling for a
             * new solver */
            newSolver = FALSE;
        } /* end of case of iteratin 2 */
    } /* end of special processing for a new solver */

    if (foundError(error)) {
      if (error == spSINGULAR) {
	int badRow, badCol;
	spWhereSingular(pDevice->matrix, &badRow, &badCol);
	printf("*****  singular at (%d,%d)\n", badRow, badCol);
      }
      pDevice->converged = FALSE;
      quitLoop = TRUE;
      continue;
    }

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhs, delta, NULL, NULL);
    solveTime += SPfrontEnd->IFseconds() - startTime;

    /* UPDATE */
    startTime = SPfrontEnd->IFseconds();
    /* Use norm reducing Newton method for DC bias solutions only. */
    if ((!pDevice->poissonOnly) && (iterationLimit > 0)
	&& (!tranAnalysis) && (pDevice->rhsNorm > 1e-1)) {
      error = TWOnewDelta(pDevice, tranAnalysis, info);
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
      pDevice->converged = TWOdeltaConverged(pDevice);
    }
    /* Check if the residual is smaller than abstol. */
    if (pDevice->converged && (!pDevice->poissonOnly)
	&& (!tranAnalysis)) {
      if (!OneCarrier) {
	TWO_rhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == N_TYPE) {
	TWONrhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == P_TYPE) {
	TWOPrhsLoad(pDevice, tranAnalysis, info);
      }
      pDevice->rhsNorm = maxNorm(rhs, size);
      if (pDevice->rhsNorm > pDevice->abstol) {
	pDevice->converged = FALSE;
      }
      if ((++timesConverged >= 2)
	  && (pDevice->rhsNorm < 1e3 * pDevice->abstol)) {
	pDevice->converged = TRUE;
      } else if (timesConverged >= 5) {
	pDevice->converged = FALSE;
	quitLoop = TRUE;
	continue;
      }
    } else if (pDevice->converged && pDevice->poissonOnly) {
      TWOQrhsLoad(pDevice);
      pDevice->rhsNorm = maxNorm(rhs, size);
      if (pDevice->rhsNorm > pDevice->abstol) {
	pDevice->converged = FALSE;
      }
      if (++timesConverged >= 5) {
	pDevice->converged = TRUE;
      }
    }
    /* Check if the carrier concentrations are negative. */
    if (pDevice->converged && (!pDevice->poissonOnly)) {
      /* Clear garbage entry since carrier-free elements reference it. */
      solution[0] = 0.0;
      for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
	pElem = pDevice->elements[eIndex];
	for (index = 0; index <= 3; index++) {
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
      if (!pDevice->converged) {
	if (!OneCarrier) {
	  TWO_rhsLoad(pDevice, tranAnalysis, info);
	} else if (OneCarrier == N_TYPE) {
	  TWONrhsLoad(pDevice, tranAnalysis, info);
	} else if (OneCarrier == P_TYPE) {
	  TWOPrhsLoad(pDevice, tranAnalysis, info);
	}
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
	poissNorm = contNorm = 0.0;
	rhs[0] = 0.0;		/* Make sure garbage entry is clear. */
	for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
	  pElem = pDevice->elements[eIndex];
	  for (index = 0; index <= 3; index++) {
	    if (pElem->evalNodes[index]) {
	      pNode = pElem->pNodes[index];
	      poissNorm = MAX(poissNorm,ABS(rhs[pNode->psiEqn]));
	      contNorm = MAX(contNorm,ABS(rhs[pNode->nEqn]));
	      contNorm = MAX(contNorm,ABS(rhs[pNode->pEqn]));
	    }
	  }
	}
	fprintf(stdout,
	    "Residual: %11.4e C/um poisson, %11.4e A/um continuity\n",
	    poissNorm * EpsNorm * VNorm * 1e-4,
	    contNorm * JNorm * LNorm * 1e-4);
      } else {
	fprintf(stdout, "Residual: %11.4e C/um poisson\n",
	    pDevice->rhsNorm * EpsNorm * VNorm * 1e-4);
      }
    }
  }
}

BOOLEAN
TWOdeltaConverged(TWOdevice *pDevice)
{
  /* This function returns a TRUE if converged else a FALSE. */
  int index;
  BOOLEAN converged = TRUE;
  double xOld, xNew, tol;

  for (index = 1; index <= pDevice->numEqns; index++) {
    xOld = pDevice->dcSolution[index];
    xNew = xOld + pDevice->dcDeltaSolution[index];
    tol = pDevice->abstol + pDevice->reltol * MAX(ABS(xOld), ABS(xNew));
    if (ABS(xOld - xNew) > tol) {
      converged = FALSE;
      break;
    }
  }
  return (converged);
}

BOOLEAN
TWOdeviceConverged(TWOdevice *pDevice)
{
  int index, eIndex;
  BOOLEAN converged = TRUE;
  double *solution = pDevice->dcSolution;
  TWOnode *pNode;
  TWOelem *pElem;
  double startTime;

  /* If the update is sufficiently small, and the carrier concentrations
   * are all positive, then return TRUE, else return FALSE.
   */

  /* CHECK CONVERGENCE */
  startTime = SPfrontEnd->IFseconds();
  if ((converged = TWOdeltaConverged(pDevice)) == TRUE) {
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      for (index = 0; index <= 3; index++) {
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

void
TWOresetJacobian(TWOdevice *pDevice)
{
  int error;
 

  if (!OneCarrier) {
    TWO_jacLoad(pDevice);
  } else if (OneCarrier == N_TYPE) {
    TWONjacLoad(pDevice);
  } else if (OneCarrier == P_TYPE) {
    TWOPjacLoad(pDevice);
  } else {
    printf("TWOresetJacobian: unknown carrier type\n");
    exit(-1);
  }
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
TWOstoreNeutralGuess(TWOdevice *pDevice)
{
  int nIndex, eIndex;
  TWOelem *pElem;
  TWOnode *pNode;
  double nie, conc, absConc, refPsi, psi, ni, pi, sign;

  /* assign the initial guess for Poisson's equation */
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    refPsi = pElem->matlInfo->refPsi;
    if (pElem->elemType == INSULATOR) {
      for (nIndex = 0; nIndex <= 3; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  if (pNode->nodeType == CONTACT) {
	    /*
	     * a metal contact to insulator domain so use work function
	     * difference as the value of psi
	     */
	    pNode->psi = RefPsi - pNode->eaff;
	  } else {
	    pNode->psi = refPsi;
	  }
	}
      }
    }
    if (pElem->elemType == SEMICON) {
      for (nIndex = 0; nIndex <= 3; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  /* silicon nodes */
	  nie = pNode->nie;
	  conc = pNode->netConc / pNode->nie;
	  psi = 0.0;
	  ni = nie;
	  pi = nie;
	  sign = SGN(conc);
	  absConc = ABS(conc);
	  if (conc != 0.0) {
	    psi = sign * log(0.5 * absConc
		+ sqrt(1.0 + 0.25 * absConc * absConc));
	    ni = nie * exp(psi);
	    pi = nie * exp(-psi);
	  }
	  pNode->psi = refPsi + psi;
	  pNode->nConc = ni;
	  pNode->pConc = pi;
	  /* store the initial guess in the dc solution vector */
	  if (pNode->nodeType != CONTACT) {
	    pDevice->dcSolution[pNode->poiEqn] = pNode->psi;
	  }
	}
      }
    }
  }
}

/* computing the equilibrium solution; solution of Poisson's eqn */
/* the solution is used as an initial starting point for bias conditions */
int TWOequilSolve(TWOdevice *pDevice)
{
    BOOLEAN newSolver = FALSE;
    int error;
    int nIndex, eIndex;
    TWOelem *pElem;
    TWOnode *pNode;
    double startTime, setupTime, miscTime;

    setupTime = miscTime = 0.0;

    /* SETUP */
    startTime = SPfrontEnd->IFseconds();

    /* Set up pDevice to compute the equilibrium solution. If the solver
     * is for bias, the arrays must be freed and allocated to the correct
     * sizes for an equilibrium solution; if it is a new solver, they are
     * only allocated; and if already an equilibrium solve, nothing
     * needs to be done */
    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch (pDevice->solverType) {
        case SLV_SMSIG:
        case SLV_BIAS:
            /* Free memory allocated for the bias solution */
            FREE(pDevice->dcSolution);
            FREE(pDevice->dcDeltaSolution);
            FREE(pDevice->copiedSolution);
            FREE(pDevice->rhs);
            FREE(pDevice->rhsImag);
            spDestroy(pDevice->matrix);
            /* FALLTHROUGH */
        case SLV_NONE: {
            /* Allocate memory needed for an equilibrium solution */
            const int n_dim = pDevice->dimEquil;
            const int n_eqn = n_dim - 1;
            pDevice->poissonOnly = TRUE;
            pDevice->numEqns = n_eqn;
            XCALLOC(pDevice->dcSolution, double, n_dim);
            XCALLOC(pDevice->dcDeltaSolution, double, n_dim);
            XCALLOC(pDevice->copiedSolution, double, n_dim);
            XCALLOC(pDevice->rhs, double, n_dim);
            pDevice->matrix = spCreate(n_eqn, 0, &error);
            if (error == spNO_MEMORY) {
                (void) fprintf(cp_err, "TWOequilSolve: Out of Memory\n");
                return E_NOMEM;
            }
            newSolver = TRUE;
            spSetReal(pDevice->matrix); /* set to a real matrix */
            TWOQjacBuild(pDevice);
            pDevice->numOrigEquil = spElementCount(pDevice->matrix);
            pDevice->numFillEquil = 0;
            pDevice->solverType = SLV_EQUIL;
            break;
        }
        case SLV_EQUIL: /* Nothing to do if already equilibrium solver */
            break;
        default: /* Invalid data */
            fprintf(stderr, "Panic: Unknown solver type in equil solution.\n");
            return E_PANIC;
    } /* end of switch over solve type */
    TWOstoreNeutralGuess(pDevice);
    setupTime += SPfrontEnd->IFseconds() - startTime;

  /* SOLVE */
  TWOdcSolve(pDevice, MaxIterations, newSolver, FALSE, NULL);

  /* MISCELLANEOUS */
  startTime = SPfrontEnd->IFseconds();
  if (newSolver) {
    pDevice->numFillEquil = spFillinCount(pDevice->matrix);
  }
  if (pDevice->converged) {
    TWOQcommonTerms(pDevice);

    /* save equilibrium potential */
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      for (nIndex = 0; nIndex <= 3; nIndex++) {
	if (pElem->evalNodes[nIndex]) {
	  pNode = pElem->pNodes[nIndex];
	  pNode->psi0 = pNode->psi;
	}
      }
    }
  } else {
    printf("TWOequilSolve: No Convergence\n");
  }
  miscTime += SPfrontEnd->IFseconds() - startTime;
  pDevice->pStats->setupTime[STAT_SETUP] += setupTime;
  pDevice->pStats->miscTime[STAT_SETUP] += miscTime;

  return 0;
}

/* compute the solution under an applied bias */
/* the equilibrium solution is taken as an initial guess */
void
TWObiasSolve(TWOdevice *pDevice, int iterationLimit, BOOLEAN tranAnalysis, 
             TWOtranInfo *info)
{
  BOOLEAN newSolver = FALSE;
  int error;
  int index, eIndex;
  TWOelem *pElem;
  TWOnode *pNode;
  double refPsi;
  double startTime, setupTime, miscTime;

  setupTime = miscTime = 0.0;

  /* SETUP */
  startTime = SPfrontEnd->IFseconds();
    /* Set up for solving for bias */
    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch (pDevice->solverType) {
    case SLV_EQUIL:
        /* free up the vectors allocated in the equilibrium solution */
        FREE(pDevice->dcSolution);
        FREE(pDevice->dcDeltaSolution);
        FREE(pDevice->copiedSolution);
        FREE(pDevice->rhs);
        spDestroy(pDevice->matrix);
        /* FALLTHROUGH */
  case SLV_NONE:
    /* Set up for bias */
    pDevice->poissonOnly = FALSE;
    pDevice->numEqns = pDevice->dimBias - 1;
    XCALLOC(pDevice->dcSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->dcDeltaSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->copiedSolution, double, pDevice->dimBias);
    XCALLOC(pDevice->rhs, double, pDevice->dimBias);
    XCALLOC(pDevice->rhsImag, double, pDevice->dimBias);
    pDevice->matrix = spCreate(pDevice->numEqns, 1, &error);
    if (error == spNO_MEMORY) {
      printf("TWObiasSolve: Out of Memory\n");
      exit(-1);
    }
    newSolver = TRUE;
    if (!OneCarrier) {
      TWO_jacBuild(pDevice);
    } else if (OneCarrier == N_TYPE) {
      TWONjacBuild(pDevice);
    } else if (OneCarrier == P_TYPE) {
      TWOPjacBuild(pDevice);
    }
    pDevice->numOrigBias = spElementCount(pDevice->matrix);
    pDevice->numFillBias = 0;
    TWOstoreInitialGuess(pDevice);
        /* FALLTHROUGH */
  case SLV_SMSIG:
    spSetReal(pDevice->matrix);
    pDevice->solverType = SLV_BIAS;
    break;
  case SLV_BIAS:
    break;
  default:
    fprintf(stderr, "Panic: Unknown solver type in bias solution.\n");
    exit(-1);
    break;
  }
  setupTime += SPfrontEnd->IFseconds() - startTime;

  /* SOLVE */
  TWOdcSolve(pDevice, iterationLimit, newSolver, tranAnalysis, info);

  /* MISCELLANEOUS */
  startTime = SPfrontEnd->IFseconds();
  if (newSolver) {
    pDevice->numFillBias = spFillinCount(pDevice->matrix);
  }
  if ((!pDevice->converged) && iterationLimit > 1) {
    printf("TWObiasSolve: No Convergence\n");
  } else if (pDevice->converged) {
    /* update the nodal quantities */
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      refPsi = pElem->matlInfo->refPsi;
      for (index = 0; index <= 3; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    pNode->psi = pDevice->dcSolution[pNode->psiEqn];
	    if (pElem->elemType == SEMICON) {
	      if (!OneCarrier) {
		pNode->nConc = pDevice->dcSolution[pNode->nEqn];
		pNode->pConc = pDevice->dcSolution[pNode->pEqn];
	      } else if (OneCarrier == N_TYPE) {
		pNode->nConc = pDevice->dcSolution[pNode->nEqn];
		pNode->pConc = pNode->nie * exp(-pNode->psi + refPsi);
	      } else if (OneCarrier == P_TYPE) {
		pNode->pConc = pDevice->dcSolution[pNode->pEqn];
		pNode->nConc = pNode->nie * exp(pNode->psi - refPsi);
	      }
	    }
	  }
	}
      }
    }

    /* update the current terms */
    if (!OneCarrier) {
      TWO_commonTerms(pDevice, FALSE, tranAnalysis, info);
    } else if (OneCarrier == N_TYPE) {
      TWONcommonTerms(pDevice, FALSE, tranAnalysis, info);
    } else if (OneCarrier == P_TYPE) {
      TWOPcommonTerms(pDevice, FALSE, tranAnalysis, info);
    }
  } else if (iterationLimit <= 1) {
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      refPsi = pElem->matlInfo->refPsi;
      for (index = 0; index <= 3; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    pNode->psi = pDevice->dcSolution[pNode->psiEqn];
	    pDevice->devState0 [pNode->nodePsi] = pNode->psi;
	    if (pElem->elemType == SEMICON) {
	      if (!OneCarrier) {
		pNode->nConc = pDevice->dcSolution[pNode->nEqn];
		pNode->pConc = pDevice->dcSolution[pNode->pEqn];
	      } else if (OneCarrier == N_TYPE) {
		pNode->nConc = pDevice->dcSolution[pNode->nEqn];
		pNode->pConc = pNode->nie * exp(-pNode->psi + refPsi);
	      } else if (OneCarrier == P_TYPE) {
		pNode->pConc = pDevice->dcSolution[pNode->pEqn];
		pNode->nConc = pNode->nie * exp(pNode->psi - refPsi);
	      }
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
TWOstoreEquilibGuess(TWOdevice *pDevice)
{
  int nIndex, eIndex;
  double *solution = pDevice->dcSolution;
  double refPsi;
  TWOelem *pElem;
  TWOnode *pNode;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    refPsi = pElem->matlInfo->refPsi;
    for (nIndex = 0; nIndex <= 3; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pNode->nodeType != CONTACT) {
	  pDevice->dcSolution[pNode->psiEqn] = pNode->psi0;
	  if (pElem->elemType == SEMICON) {
	    if (!OneCarrier) {
	      solution[pNode->nEqn] = pNode->nie * exp(pNode->psi0 - refPsi);
	      solution[pNode->pEqn] = pNode->nie * exp(-pNode->psi0 + refPsi);
	    } else if (OneCarrier == N_TYPE) {
	      solution[pNode->nEqn] = pNode->nie * exp(pNode->psi0 - refPsi);
	    } else if (OneCarrier == P_TYPE) {
	      solution[pNode->pEqn] = pNode->nie * exp(-pNode->psi0 + refPsi);
	    }
	  }
	}
      }
    }
  }
}

/* Copy the device's internal state into the solution vector. */
void
TWOstoreInitialGuess(TWOdevice *pDevice)
{
  int nIndex, eIndex;
  double *solution = pDevice->dcSolution;
  TWOelem *pElem;
  TWOnode *pNode;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (nIndex = 0; nIndex <= 3; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if (pNode->nodeType != CONTACT) {
	  solution[pNode->psiEqn] = pNode->psi;
	  if (pElem->elemType == SEMICON) {
	    if (!OneCarrier) {
	      solution[pNode->nEqn] = pNode->nConc;
	      solution[pNode->pEqn] = pNode->pConc;
	    } else if (OneCarrier == N_TYPE) {
	      solution[pNode->nEqn] = pNode->nConc;
	    } else if (OneCarrier == P_TYPE) {
	      solution[pNode->pEqn] = pNode->pConc;
	    }
	  }
	}
      }
    }
  }
}


/*
 * function computeNewDelta computes an acceptable delta
 */

void
oldTWOnewDelta(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  int index;
  double newNorm, fib, lambda, fibn, fibp;
  BOOLEAN acceptable = FALSE;
  lambda = 1.0;
  fibn = 1.0;
  fibp = 1.0;

  /*
   * copy the contents of dcSolution into copiedSolution and modify
   * dcSolution by adding the deltaSolution
   */

  for (index = 1; index <= pDevice->numEqns; ++index) {
    pDevice->copiedSolution[index] = pDevice->dcSolution[index];
    pDevice->dcSolution[index] += pDevice->dcDeltaSolution[index];
  }

  /* compute L2 norm of the deltaX vector */
  pDevice->rhsNorm = l2Norm(pDevice->dcDeltaSolution, pDevice->numEqns);

  if (pDevice->poissonOnly) {
    TWOQrhsLoad(pDevice);
  } else if (!OneCarrier) {
    TWO_rhsLoad(pDevice, tranAnalysis, info);
  } else if (OneCarrier == N_TYPE) {
    TWONrhsLoad(pDevice, tranAnalysis, info);
  } else if (OneCarrier == P_TYPE) {
    TWOPrhsLoad(pDevice, tranAnalysis, info);
  }
  newNorm = TWOnuNorm(pDevice);
  if (newNorm <= pDevice->rhsNorm) {
    acceptable = TRUE;
  } else {
    /* chop the step size so that deltax is acceptable */
    for (; !acceptable;) {
      fib = fibp;
      fibp = fibn;
      fibn += fib;
      lambda *= (fibp / fibn);

      for (index = 1; index <= pDevice->numEqns; index++) {
	pDevice->dcSolution[index] = pDevice->copiedSolution[index]
	    + lambda * pDevice->dcDeltaSolution[index];
      }
      if (pDevice->poissonOnly) {
	TWOQrhsLoad(pDevice);
      } else if (!OneCarrier) {
	TWO_rhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == N_TYPE) {
	TWONrhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == P_TYPE) {
	TWOPrhsLoad(pDevice, tranAnalysis, info);
      }
      newNorm = TWOnuNorm(pDevice);
      if (newNorm <= pDevice->rhsNorm) {
	acceptable = TRUE;
      }
    }
  }
  /* restore the previous dcSolution and the new deltaSolution */
  pDevice->rhsNorm = newNorm;
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->dcSolution[index] = pDevice->copiedSolution[index];
    pDevice->dcDeltaSolution[index] *= lambda;
  }
}



int
TWOnewDelta(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  int index, iterNum = 0;
  double newNorm;
  double fib, lambda, fibn, fibp;
  BOOLEAN acceptable = FALSE, error = FALSE;

  lambda = 1.0;
  fibn = 1.0;
  fibp = 1.0;

  /*
   * copy the contents of dcSolution into copiedSolution and modify
   * dcSolution by adding the deltaSolution
   */

  for (index = 1; index <= pDevice->numEqns; ++index) {
    pDevice->copiedSolution[index] = pDevice->dcSolution[index];
    pDevice->dcSolution[index] += pDevice->dcDeltaSolution[index];
  }

  if (pDevice->poissonOnly) {
    TWOQrhsLoad(pDevice);
  } else if (!OneCarrier) {
    TWO_rhsLoad(pDevice, tranAnalysis, info);
  } else if (OneCarrier == N_TYPE) {
    TWONrhsLoad(pDevice, tranAnalysis, info);
  } else if (OneCarrier == P_TYPE) {
    TWOPrhsLoad(pDevice, tranAnalysis, info);
  }
  newNorm = maxNorm(pDevice->rhs, pDevice->numEqns);
  if (pDevice->rhsNorm <= pDevice->abstol) {
    lambda = 0.0;
    newNorm = pDevice->rhsNorm;
  } else if (newNorm < pDevice->rhsNorm) {
    acceptable = TRUE;
  } else {
    /* chop the step size so that deltax is acceptable */
    if (TWOdcDebug) {
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
	TWOQrhsLoad(pDevice);
      } else if (!OneCarrier) {
	TWO_rhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == N_TYPE) {
	TWONrhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == P_TYPE) {
	TWOPrhsLoad(pDevice, tranAnalysis, info);
      }
      newNorm = maxNorm(pDevice->rhs, pDevice->numEqns);
      if (error) {
	break;
      }
      if (newNorm <= pDevice->rhsNorm) {
	acceptable = TRUE;
      }
      if (TWOdcDebug) {
	fprintf(stdout, "          %11.4e  %11.4e\n",
	    newNorm, lambda);
      }
    }
  }
  /* restore the previous dcSolution and the new deltaSolution */
  pDevice->rhsNorm = newNorm;
  for (index = 1; index <= pDevice->numEqns; index++) {
    pDevice->dcSolution[index] = pDevice->copiedSolution[index];
    pDevice->dcDeltaSolution[index] *= lambda;
  }
  return (error);
}


/* Predict the values of the internal variables at the next timepoint. */
/* Needed for Predictor-Corrector LTE estimation */
void
TWOpredict(TWOdevice *pDevice, TWOtranInfo *info)
{
  int nIndex, eIndex;
  TWOnode *pNode;
  TWOelem *pElem;
  double startTime, miscTime = 0.0;

  /* TRANSIENT MISC */
  startTime = SPfrontEnd->IFseconds();
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (nIndex = 0; nIndex <= 3; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	pNode->psi = pDevice->devState1 [pNode->nodePsi];
	if ((pElem->elemType == SEMICON) && (pNode->nodeType != CONTACT)) {
	  if (!OneCarrier) {
	    pNode->nPred = predict(pDevice->devStates, info, pNode->nodeN);
	    pNode->pPred = predict(pDevice->devStates, info, pNode->nodeP);
	  } else if (OneCarrier == N_TYPE) {
	    pNode->nPred = predict(pDevice->devStates, info, pNode->nodeN);
	    pNode->pPred = pDevice->devState1 [pNode->nodeP];
	  } else if (OneCarrier == P_TYPE) {
	    pNode->pPred = predict(pDevice->devStates, info, pNode->nodeP);
	    pNode->nPred = pDevice->devState1 [pNode->nodeN];
	  }
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
TWOtrunc(TWOdevice *pDevice, TWOtranInfo *info, double delta)
{
  int nIndex, eIndex;
  TWOelem *pElem;
  TWOnode *pNode;
  double tolN, tolP, lte, relError, temp;
  double lteCoeff = info->lteCoeff;
  double mult = 10.0;
  double reltol;
  double startTime, lteTime = 0.0;

  /* TRANSIENT LTE */
  startTime = SPfrontEnd->IFseconds();

  relError = 0.0;
  reltol = pDevice->reltol * mult;

  /* need to get the predictor for the current order */
  /* the scheme currently used is very dumb. need to fix later */
  computePredCoeff(info->method, info->order, info->predCoeff, info->delta);

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (nIndex = 0; nIndex <= 3; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	if ((pElem->elemType == SEMICON) && (pNode->nodeType != CONTACT)) {
	  if (!OneCarrier) {
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
	  } else if (OneCarrier == N_TYPE) {
	    tolN = pDevice->abstol + reltol * ABS(pNode->nConc);
	    pNode->nPred = predict(pDevice->devStates, info, pNode->nodeN);
	    lte = lteCoeff * (pNode->nConc - pNode->nPred);
	    temp = lte / tolN;
	    relError += temp * temp;
	  } else if (OneCarrier == P_TYPE) {
	    tolP = pDevice->abstol + reltol * ABS(pNode->pConc);
	    pNode->pPred = predict(pDevice->devStates, info, pNode->nodeP);
	    lte = lteCoeff * (pNode->pConc - pNode->pPred);
	    temp = lte / tolP;
	    relError += temp * temp;
	  }
	}
      }
    }
  }

  relError = MAX(pDevice->abstol, relError);	/* make sure it is non zero */

  /* the total relative error has been calculated norm-2 squared */
  relError = sqrt(relError / pDevice->numEqns);

  /* depending on the order of the integration method compute new delta */
  temp = delta / pow(relError, 1.0 / (info->order + 1));

  lteTime += SPfrontEnd->IFseconds() - startTime;
  pDevice->pStats->lteTime += lteTime;

  /* return the new delta (stored as temp) */
  return (temp);
}

/* Save info from state table into the internal state. */
void
TWOsaveState(TWOdevice *pDevice)
{
  int nIndex, eIndex;
  TWOnode *pNode;
  TWOelem *pElem;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (nIndex = 0; nIndex <= 3; nIndex++) {
      if (pElem->evalNodes[nIndex]) {
	pNode = pElem->pNodes[nIndex];
	pNode->psi = pDevice->devState1 [pNode->nodePsi];
	if ((pElem->elemType == SEMICON) && (pNode->nodeType != CONTACT)) {
	  pNode->nConc = pDevice->devState1 [pNode->nodeN];
	  pNode->pConc = pDevice->devState1 [pNode->nodeP];
	}
      }
    }
  }
}

/*
 * A function that checks convergence based on the convergence of the quasi
 * Fermi levels. This should be better since we are looking at potentials in
 * all (psi, phin, phip)
 */
BOOLEAN
TWOpsiDeltaConverged(TWOdevice *pDevice)
{
  int index, nIndex, eIndex;
  TWOnode *pNode;
  TWOelem *pElem;
  double xOld, xNew, xDelta, tol;
  double psi, newPsi, nConc, pConc, newN, newP;
  double phiN, phiP, newPhiN, newPhiP;
  BOOLEAN converged = TRUE;

  /* equilibrium solution */
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
  /* bias solution. check convergence on psi, phin, phip */
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (nIndex = 0; nIndex <= 3; nIndex++) {
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
	if ((pElem->elemType == SEMICON) && (pNode->nodeType != CONTACT)) {
	  psi = pDevice->dcSolution[pNode->psiEqn];
	  nConc = pDevice->dcSolution[pNode->nEqn];
	  pConc = pDevice->dcSolution[pNode->pEqn];
	  newPsi = psi + pDevice->dcDeltaSolution[pNode->psiEqn];
	  newN = nConc + pDevice->dcDeltaSolution[pNode->nEqn];
	  newP = pConc + pDevice->dcDeltaSolution[pNode->pEqn];
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


/* Function to compute Nu norm of the rhs vector. */
/* nu-Norm calculation based upon work done at Stanford. */
double
TWOnuNorm(TWOdevice *pDevice)
{
  double norm = 0.0;
  double temp;
  int index;

  /* the LU Decomposed matrix is available. use it to calculate x */

  spSolve(pDevice->matrix, pDevice->rhs, pDevice->rhsImag,
      NULL, NULL);

  /* the solution is in the rhsImag vector */
  /* compute L2-norm of the rhsImag vector */

  for (index = 1; index <= pDevice->numEqns; index++) {
    temp = pDevice->rhsImag[index];
    norm += temp * temp;
  }
  norm = sqrt(norm);
  return (norm);
}

/*
 * Check for numerical errors in the Jacobian.  Useful debugging aid when new
 * models are being incorporated.
 */
void
TWOjacCheck(TWOdevice *pDevice, BOOLEAN tranAnalysis, TWOtranInfo *info)
{
  int index, rIndex;
  double del, diff, tol, *dptr;

  if (TWOjacDebug) {
    if (!OneCarrier) {
      TWO_sysLoad(pDevice, tranAnalysis, info);
    } else if (OneCarrier == N_TYPE) {
      TWONsysLoad(pDevice, tranAnalysis, info);
    } else if (OneCarrier == P_TYPE) {
      TWOPsysLoad(pDevice, tranAnalysis, info);
    }
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
      if (!OneCarrier) {
	TWO_rhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == N_TYPE) {
	TWONrhsLoad(pDevice, tranAnalysis, info);
      } else if (OneCarrier == P_TYPE) {
	TWOPrhsLoad(pDevice, tranAnalysis, info);
      }
      pDevice->dcSolution[index] = pDevice->copiedSolution[index];
      for (rIndex = 1; rIndex <= pDevice->numEqns; rIndex++) {
	diff = (pDevice->rhsImag[rIndex] - pDevice->rhs[rIndex]) / del;
	dptr = spFindElement(pDevice->matrix, rIndex, index);
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
