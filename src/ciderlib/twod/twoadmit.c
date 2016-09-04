/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/* Functions to compute the ac admittances of a device. */

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/numconst.h"
#include "ngspice/twodev.h"
#include "ngspice/twomesh.h"
#include "ngspice/complex.h"
#include "ngspice/spmatrix.h"
#include "ngspice/bool.h"
#include "ngspice/macros.h"
#include "ngspice/ifsim.h"
#include "twoddefs.h"  
#include "twodext.h"
#include "ngspice/cidersupt.h"

extern IFfrontEnd *SPfrontEnd;

/* 
 * mmhhh this may cause troubles
 * Paolo Nenzi 2002
 */
 SPcomplex yTotal;

int
NUMD2admittance(TWOdevice *pDevice, double omega, SPcomplex *yd)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double dxdy;
  double *solnReal, *solnImag;
  double *rhsReal, *rhsImag;
  SPcomplex yAc, cOmega, *y;
  BOOLEAN deltaVContact = FALSE;
  BOOLEAN SORFailed;
  double startTime;

  /* Each time we call this counts as one AC iteration. */
  pDevice->pStats->numIters[STAT_AC] += 1;

  /*
   * change context names of solution vectors for ac analysis dcDeltaSolution
   * stores the real part and copiedSolution stores the imaginary part of the
   * ac solution vector
   */
  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  omega *= TNorm;
  CMPLX_ASSIGN_VALUE(cOmega, 0.0, omega);

  if ((AcAnalysisMethod == SOR) || (AcAnalysisMethod == SOR_ONLY)) {
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    /* zero the rhsImag */
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    /* store the new rhs vector */
    storeNewRhs(pDevice, pDevice->pLastContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
    if (SORFailed && AcAnalysisMethod == SOR) {
      AcAnalysisMethod = DIRECT;
      printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	  omega / (2 * M_PI * TNorm) );
    } else if (SORFailed) {	/* Told to only do SOR, so give up. */
      printf("SOR failed at %g Hz, returning null admittance.\n",
	  omega / (2 * M_PI * TNorm) );
      CMPLX_ASSIGN_VALUE(*yd, 0.0, 0.0);
      return (AcAnalysisMethod);
    }
  }
  if (AcAnalysisMethod == DIRECT) {
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    /* solve the system of equations directly */
    if (!OneCarrier) {
      TWO_jacLoad(pDevice);
    } else if (OneCarrier == N_TYPE) {
      TWONjacLoad(pDevice);
    } else if (OneCarrier == P_TYPE) {
      TWOPjacLoad(pDevice);
    }
    storeNewRhs(pDevice, pDevice->pLastContact);

    spSetComplex(pDevice->matrix);
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      if (pElem->elemType == SEMICON) {
	dxdy = 0.25 * pElem->dx * pElem->dy;
	for (index = 0; index <= 3; index++) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    if (!OneCarrier) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    } else if (OneCarrier == N_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	    } else if (OneCarrier == P_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    }
	  }
	}
      }
    }
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR */
    startTime = SPfrontEnd->IFseconds();
    spFactor(pDevice->matrix);
    pDevice->pStats->factorTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
  }
  /* MISC */
  startTime = SPfrontEnd->IFseconds();
  y = contactAdmittance(pDevice, pDevice->pFirstContact, deltaVContact,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc, -y->real, -y->imag);
  CMPLX_ASSIGN(*yd, yAc);
  CMPLX_MULT_SELF_SCALAR(*yd, GNorm * pDevice->width * LNorm);
  pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

  return (AcAnalysisMethod);
}


int
NBJT2admittance(TWOdevice *pDevice, double omega, SPcomplex *yIeVce, 
                SPcomplex *yIcVce, SPcomplex *yIeVbe, SPcomplex *yIcVbe)
{
  TWOcontact *pEmitContact = pDevice->pLastContact;
  TWOcontact *pColContact = pDevice->pFirstContact;
  TWOcontact *pBaseContact = pDevice->pFirstContact->next;
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double width = pDevice->width;
  double dxdy;
  double *solnReal, *solnImag;
  double *rhsReal, *rhsImag;
  BOOLEAN SORFailed;
  SPcomplex *y;
  SPcomplex pIeVce, pIcVce, pIeVbe, pIcVbe;
  SPcomplex cOmega;
  double startTime;

  /* Each time we call this counts as one AC iteration. */
  pDevice->pStats->numIters[STAT_AC] += 1;

  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  omega *= TNorm;
  CMPLX_ASSIGN_VALUE(cOmega, 0.0, omega);
  CMPLX_ASSIGN_VALUE(pIeVce, NAN, NAN);
  CMPLX_ASSIGN_VALUE(pIcVce, NAN, NAN);

  if ((AcAnalysisMethod == SOR) || (AcAnalysisMethod == SOR_ONLY)) {
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    /* zero the rhs before loading in the new rhs */
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pColContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
    if (SORFailed && AcAnalysisMethod == SOR) {
      AcAnalysisMethod = DIRECT;
      printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	  omega / (2 * M_PI * TNorm) );
    } else if (SORFailed) {	/* Told to only do SOR, so give up. */
      printf("SOR failed at %g Hz, returning null admittance.\n",
	  omega / (2 * M_PI * TNorm) );
      CMPLX_ASSIGN_VALUE(*yIeVce, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(*yIcVce, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(*yIeVbe, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(*yIcVbe, 0.0, 0.0);
      return (AcAnalysisMethod);
    } else {
      /* MISC */
      startTime = SPfrontEnd->IFseconds();
      y = contactAdmittance(pDevice, pEmitContact, FALSE,
	  solnReal, solnImag, &cOmega);
      CMPLX_ASSIGN_VALUE(pIeVce, y->real, y->imag);
      y = contactAdmittance(pDevice, pColContact, TRUE,
	  solnReal, solnImag, &cOmega);
      CMPLX_ASSIGN_VALUE(pIcVce, y->real, y->imag);
      pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

      /* LOAD */
      startTime = SPfrontEnd->IFseconds();
      /* load in the base contribution to the rhs */
      for (index = 1; index <= pDevice->numEqns; index++) {
	rhsImag[index] = 0.0;
      }
      storeNewRhs(pDevice, pBaseContact);
      pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

      /* SOLVE */
      startTime = SPfrontEnd->IFseconds();
      SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
      pDevice->pStats->solveTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
      if (SORFailed && AcAnalysisMethod == SOR) {
	AcAnalysisMethod = DIRECT;
	printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	    omega / (2 * M_PI * TNorm) );
      } else if (SORFailed) {	/* Told to only do SOR, so give up. */
	printf("SOR failed at %g Hz, returning null admittance.\n",
	    omega / (2 * M_PI * TNorm) );
	CMPLX_ASSIGN_VALUE(*yIeVce, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(*yIcVce, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(*yIeVbe, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(*yIcVbe, 0.0, 0.0);
	return (AcAnalysisMethod);
      }
    }
  }
  if (AcAnalysisMethod == DIRECT) {
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    /* solve the system of equations directly */
    if (!OneCarrier) {
      TWO_jacLoad(pDevice);
    } else if (OneCarrier == N_TYPE) {
      TWONjacLoad(pDevice);
    } else if (OneCarrier == P_TYPE) {
      TWOPjacLoad(pDevice);
    }
    storeNewRhs(pDevice, pColContact);
    spSetComplex(pDevice->matrix);
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      if (pElem->elemType == SEMICON) {
	dxdy = 0.25 * pElem->dx * pElem->dy;
	for (index = 0; index <= 3; index++) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    if (!OneCarrier) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    } else if (OneCarrier == N_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	    } else if (OneCarrier == P_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    }
	  }
	}
      }
    }
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR */
    startTime = SPfrontEnd->IFseconds();
    spFactor(pDevice->matrix);
    pDevice->pStats->factorTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* MISC */
    startTime = SPfrontEnd->IFseconds();
    y = contactAdmittance(pDevice, pEmitContact, FALSE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(pIeVce, y->real, y->imag);
    y = contactAdmittance(pDevice, pColContact, TRUE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(pIcVce, y->real, y->imag);
    pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pBaseContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR: already done, no need to repeat. */

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
  }
  /* MISC */
  startTime = SPfrontEnd->IFseconds();
  y = contactAdmittance(pDevice, pEmitContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIeVbe, y->real, y->imag);
  y = contactAdmittance(pDevice, pColContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIcVbe, y->real, y->imag);

  CMPLX_ASSIGN(*yIeVce, pIeVce);
  CMPLX_ASSIGN(*yIeVbe, pIeVbe);
  CMPLX_ASSIGN(*yIcVce, pIcVce);
  CMPLX_ASSIGN(*yIcVbe, pIcVbe);
  CMPLX_MULT_SELF_SCALAR(*yIeVce, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIeVbe, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIcVce, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIcVbe, GNorm * width * LNorm);
  pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

  return (AcAnalysisMethod);
}

int
NUMOSadmittance(TWOdevice *pDevice, double omega, struct mosAdmittances *yAc)
{
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
/*  TWOcontact *pBContact = pDevice->pLastContact; */
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double width = pDevice->width;
  double dxdy;
  double *solnReal, *solnImag;
  double *rhsReal, *rhsImag;
  BOOLEAN SORFailed;
  SPcomplex *y, cOmega;
  double startTime;

  /* Each time we call this counts as one AC iteration. */
  pDevice->pStats->numIters[STAT_AC] += 1;

  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  omega *= TNorm;
  CMPLX_ASSIGN_VALUE(cOmega, 0.0, omega);

  if ((AcAnalysisMethod == SOR) || (AcAnalysisMethod == SOR_ONLY)) {
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    /* zero the rhs before loading in the new rhs */
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pDContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
    if (SORFailed && AcAnalysisMethod == SOR) {
      AcAnalysisMethod = DIRECT;
      printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	  omega / (2 * M_PI * TNorm) );
    } else if (SORFailed) {	/* Told to only do SOR, so give up. */
      printf("SOR failed at %g Hz, returning null admittance.\n",
	  omega / (2 * M_PI * TNorm) );
      CMPLX_ASSIGN_VALUE(yAc->yIdVdb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIdVsb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIdVgb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIsVdb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIsVsb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIsVgb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIgVdb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIgVsb, 0.0, 0.0);
      CMPLX_ASSIGN_VALUE(yAc->yIgVgb, 0.0, 0.0);
      return (AcAnalysisMethod);
    } else {
      /* MISC */
      startTime = SPfrontEnd->IFseconds();
      y = contactAdmittance(pDevice, pDContact, TRUE,
	  solnReal, solnImag, &cOmega);
      CMPLX_ASSIGN_VALUE(yAc->yIdVdb, y->real, y->imag);
      y = contactAdmittance(pDevice, pSContact, FALSE,
	  solnReal, solnImag, &cOmega);
      CMPLX_ASSIGN_VALUE(yAc->yIsVdb, y->real, y->imag);
      y = GateTypeAdmittance(pDevice, pGContact, FALSE,
	  solnReal, solnImag, &cOmega);
      CMPLX_ASSIGN_VALUE(yAc->yIgVdb, y->real, y->imag);
      pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

      /* LOAD */
      startTime = SPfrontEnd->IFseconds();
      /* load in the source contribution to the rhs */
      for (index = 1; index <= pDevice->numEqns; index++) {
	rhsImag[index] = 0.0;
      }
      storeNewRhs(pDevice, pSContact);
      pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

      /* SOLVE */
      startTime = SPfrontEnd->IFseconds();
      SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
      pDevice->pStats->solveTime[STAT_AC] +=
	  SPfrontEnd->IFseconds() - startTime;
      if (SORFailed && AcAnalysisMethod == SOR) {
	AcAnalysisMethod = DIRECT;
	printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	    omega / (2 * M_PI * TNorm) );
      } else if (SORFailed) {	/* Told to only do SOR, so give up. */
	printf("SOR failed at %g Hz, returning null admittance.\n",
	    omega / (2 * M_PI * TNorm) );
	CMPLX_ASSIGN_VALUE(yAc->yIdVdb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIdVsb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIdVgb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIsVdb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIsVsb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIsVgb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIgVdb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIgVsb, 0.0, 0.0);
	CMPLX_ASSIGN_VALUE(yAc->yIgVgb, 0.0, 0.0);
	return (AcAnalysisMethod);
      } else {
	/* MISC */
	startTime = SPfrontEnd->IFseconds();
	y = contactAdmittance(pDevice, pDContact, FALSE,
	    solnReal, solnImag, &cOmega);
	CMPLX_ASSIGN_VALUE(yAc->yIdVsb, y->real, y->imag);
	y = contactAdmittance(pDevice, pSContact, TRUE,
	    solnReal, solnImag, &cOmega);
	CMPLX_ASSIGN_VALUE(yAc->yIsVsb, y->real, y->imag);
	y = GateTypeAdmittance(pDevice, pGContact, FALSE,
	    solnReal, solnImag, &cOmega);
	CMPLX_ASSIGN_VALUE(yAc->yIgVsb, y->real, y->imag);
	pDevice->pStats->miscTime[STAT_AC] +=
	    SPfrontEnd->IFseconds() - startTime;

	/* LOAD */
	startTime = SPfrontEnd->IFseconds();
	/* load in the gate contribution to the rhs */
	for (index = 1; index <= pDevice->numEqns; index++) {
	  rhsImag[index] = 0.0;
	}
	storeNewRhs(pDevice, pGContact);
	pDevice->pStats->loadTime[STAT_AC] +=
	    SPfrontEnd->IFseconds() - startTime;

	/* SOLVE */
	startTime = SPfrontEnd->IFseconds();
	SORFailed = TWOsorSolve(pDevice, solnReal, solnImag, omega);
	pDevice->pStats->solveTime[STAT_AC] +=
	    SPfrontEnd->IFseconds() - startTime;
	if (SORFailed && AcAnalysisMethod == SOR) {
	  AcAnalysisMethod = DIRECT;
	  printf("SOR failed at %g Hz, switching to direct-method ac analysis.\n",
	      omega / (2 * M_PI * TNorm) );
	} else if (SORFailed) {	/* Told to only do SOR, so give up. */
	  printf("SOR failed at %g Hz, returning null admittance.\n",
	      omega / (2 * M_PI * TNorm) );
	  CMPLX_ASSIGN_VALUE(yAc->yIdVdb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIdVsb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIdVgb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIsVdb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIsVsb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIsVgb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIgVdb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIgVsb, 0.0, 0.0);
	  CMPLX_ASSIGN_VALUE(yAc->yIgVgb, 0.0, 0.0);
	  return (AcAnalysisMethod);
	}
      }
    }
  }
  if (AcAnalysisMethod == DIRECT) {
    /* solve the system of equations directly */
    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pDContact);

    /* Need to load & factor jacobian once. */
    if (!OneCarrier) {
      TWO_jacLoad(pDevice);
    } else if (OneCarrier == N_TYPE) {
      TWONjacLoad(pDevice);
    } else if (OneCarrier == P_TYPE) {
      TWOPjacLoad(pDevice);
    }
    spSetComplex(pDevice->matrix);
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      if (pElem->elemType == SEMICON) {
	dxdy = 0.25 * pElem->dx * pElem->dy;
	for (index = 0; index <= 3; index++) {
	  pNode = pElem->pNodes[index];
	  if (pNode->nodeType != CONTACT) {
	    if (!OneCarrier) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    } else if (OneCarrier == N_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fNN, 0.0, -dxdy * omega);
	    } else if (OneCarrier == P_TYPE) {
	      spADD_COMPLEX_ELEMENT(pNode->fPP, 0.0, dxdy * omega);
	    }
	  }
	}
      }
    }
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR */
    startTime = SPfrontEnd->IFseconds();
    spFactor(pDevice->matrix);
    pDevice->pStats->factorTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* MISC */
    startTime = SPfrontEnd->IFseconds();
    y = contactAdmittance(pDevice, pDContact, TRUE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIdVdb, y->real, y->imag);
    y = contactAdmittance(pDevice, pSContact, FALSE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIsVdb, y->real, y->imag);
    y = GateTypeAdmittance(pDevice, pGContact, FALSE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIgVdb, y->real, y->imag);
    pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pSContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR: already done, no need to repeat. */

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* MISC */
    startTime = SPfrontEnd->IFseconds();
    y = contactAdmittance(pDevice, pDContact, FALSE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIdVsb, y->real, y->imag);
    y = contactAdmittance(pDevice, pSContact, TRUE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIsVsb, y->real, y->imag);
    y = GateTypeAdmittance(pDevice, pGContact, FALSE,
	solnReal, solnImag, &cOmega);
    CMPLX_ASSIGN_VALUE(yAc->yIgVsb, y->real, y->imag);
    pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* LOAD */
    startTime = SPfrontEnd->IFseconds();
    for (index = 1; index <= pDevice->numEqns; index++) {
      rhsImag[index] = 0.0;
    }
    storeNewRhs(pDevice, pGContact);
    pDevice->pStats->loadTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

    /* FACTOR: already done, no need to repeat. */

    /* SOLVE */
    startTime = SPfrontEnd->IFseconds();
    spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
    pDevice->pStats->solveTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;
  }
  /* MISC */
  startTime = SPfrontEnd->IFseconds();
  y = contactAdmittance(pDevice, pDContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIdVgb, y->real, y->imag);
  y = contactAdmittance(pDevice, pSContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIsVgb, y->real, y->imag);
  y = GateTypeAdmittance(pDevice, pGContact, TRUE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIgVgb, y->real, y->imag);

  CMPLX_MULT_SELF_SCALAR(yAc->yIdVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIdVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIdVgb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVgb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVgb, GNorm * width * LNorm);
  pDevice->pStats->miscTime[STAT_AC] += SPfrontEnd->IFseconds() - startTime;

  return (AcAnalysisMethod);
}

BOOLEAN 
TWOsorSolve(TWOdevice *pDevice, double *xReal, double *xImag, 
            double omega)
{
  double dxdy;
  double wRelax = 1.0;		/* SOR relaxation parameter */
  double *rhsReal = pDevice->rhs;
  double *rhsSOR = pDevice->rhsImag;
  BOOLEAN SORConverged = FALSE;
  BOOLEAN SORFailed = FALSE;
  int numEqns = pDevice->numEqns;
  int iterationNum;
  int indexN, indexP;
  int index, eIndex;
  TWOnode *pNode;
  TWOelem *pElem;

  /* clear xReal and xImag arrays */
  for (index = 1; index <= numEqns; index++) {
    xReal[index] = 0.0;
    xImag[index] = 0.0;
  }

  iterationNum = 1;
  for (; (!SORConverged) &&(!SORFailed); iterationNum++) {
    for (index = 1; index <= numEqns; index++) {
      rhsSOR[index] = 0.0;
    }
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      dxdy = 0.25 * pElem->dx * pElem->dy;
      for (index = 0; index <= 3; index++) {
	pNode = pElem->pNodes[index];
	if ((pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON)) {
	  if (!OneCarrier) {
	    indexN = pNode->nEqn;
	    indexP = pNode->pEqn;
	    rhsSOR[indexN] -= dxdy * omega * xImag[indexN];
	    rhsSOR[indexP] += dxdy * omega * xImag[indexP];
	  } else if (OneCarrier == N_TYPE) {
	    indexN = pNode->nEqn;
	    rhsSOR[indexN] -= dxdy * omega * xImag[indexN];
	  } else if (OneCarrier == P_TYPE) {
	    indexP = pNode->pEqn;
	    rhsSOR[indexP] += dxdy * omega * xImag[indexP];
	  }
	}
      }
    }

    /* now add the terms from rhs to rhsImag */
    for (index = 1; index <= numEqns; index++) {
      rhsSOR[index] += rhsReal[index];
    }

    /* compute xReal(k+1). solution stored in rhsImag */
    spSolve(pDevice->matrix, rhsSOR, rhsSOR, NULL, NULL);
    /* modify solution when wRelax is not 1 */
    if (wRelax != 1) {
      for (index = 1; index <= numEqns; index++) {
	rhsSOR[index] = (1 - wRelax) * xReal[index] +
	    wRelax * rhsSOR[index];
      }
    }
    if (iterationNum > 1) {
      SORConverged = hasSORConverged(xReal, rhsSOR, numEqns);
    }
    /* copy real solution into xReal */
    for (index = 1; index <= numEqns; index++) {
      xReal[index] = rhsSOR[index];
    }

    /* now compute the imaginary part of the solution xImag */
    for (index = 1; index <= numEqns; index++) {
      rhsSOR[index] = 0.0;
    }
    for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
      pElem = pDevice->elements[eIndex];
      dxdy = 0.25 * pElem->dx * pElem->dy;
      for (index = 0; index <= 3; index++) {
	pNode = pElem->pNodes[index];
	if ((pNode->nodeType != CONTACT) && (pElem->elemType == SEMICON)) {
	  if (!OneCarrier) {
	    indexN = pNode->nEqn;
	    indexP = pNode->pEqn;
	    rhsSOR[indexN] += dxdy * omega * xReal[indexN];
	    rhsSOR[indexP] -= dxdy * omega * xReal[indexP];
	  } else if (OneCarrier == N_TYPE) {
	    indexN = pNode->nEqn;
	    rhsSOR[indexN] += dxdy * omega * xReal[indexN];
	  } else if (OneCarrier ==  P_TYPE) {
	    indexP = pNode->pEqn;
	    rhsSOR[indexP] -= dxdy * omega * xReal[indexP];
	  }
	}
      }
    }
    /* compute xImag(k+1) */
    spSolve(pDevice->matrix, rhsSOR, rhsSOR, NULL, NULL);
    /* modify solution when wRelax is not 1 */
    if (wRelax != 1) {
      for (index = 1; index <= numEqns; index++) {
	rhsSOR[index] = (1 - wRelax) * xImag[index] +
	    wRelax * rhsSOR[index];
      }
    }
    if (iterationNum > 1) {
      SORConverged = SORConverged && hasSORConverged(xImag, rhsSOR, numEqns);
    }
    /* copy imag solution into xImag */
    for (index = 1; index <= numEqns; index++) {
      xImag[index] = rhsSOR[index];
    }
    if ((iterationNum > 4) && !SORConverged) {
      SORFailed = TRUE;
    }
    if (TWOacDebug)
      printf("SOR iteration number = %d\n", iterationNum);
  }
  return (SORFailed);
}


SPcomplex *
contactAdmittance(TWOdevice *pDevice, TWOcontact *pContact, BOOLEAN delVContact,
                  double *xReal, double *xImag, SPcomplex *cOmega)
{
  TWOnode *pNode, *pHNode = NULL, *pVNode = NULL;
  TWOedge *pHEdge = NULL, *pVEdge = NULL;
  int index, i, indexPsi, indexN, indexP, numContactNodes;
  TWOelem *pElem;
  SPcomplex psiAc, nAc, pAc;
  SPcomplex prod1, prod2, sum;
  double temp;

  NG_IGNORE(pDevice);

  CMPLX_ASSIGN_VALUE(yTotal, 0.0, 0.0);

  numContactNodes = pContact->numNodes;
  for (index = 0; index < numContactNodes; index++) {
    pNode = pContact->pNodes[index];
    for (i = 0; i <= 3; i++) {
      pElem = pNode->pElems[i];
      if (pElem != NULL) {
	switch (i) {
	case 0:
	  /* the TL element */
	  pHNode = pElem->pBLNode;
	  pVNode = pElem->pTRNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  if (pElem->elemType == SEMICON) {
	    /* compute the derivatives with n,p */
	    if (pHNode->nodeType != CONTACT) {
	      indexN = pHNode->nEqn;
	      indexP = pHNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pHEdge->dJnDn);
	      CMPLX_MULT_SCALAR(prod2, pAc, pHEdge->dJpDp);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dy);
	      CMPLX_SUBT_ASSIGN(yTotal, prod1);
	    }
	    if (pVNode->nodeType != CONTACT) {
	      indexN = pVNode->nEqn;
	      indexP = pVNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pVEdge->dJnDn);
	      CMPLX_MULT_SCALAR(prod2, pAc, pVEdge->dJpDp);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dx);
	      CMPLX_SUBT_ASSIGN(yTotal, prod1);
	    }
	  }
	  break;
	case 1:
	  /* the TR element */
	  pHNode = pElem->pBRNode;
	  pVNode = pElem->pTLNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  if (pElem->elemType == SEMICON) {
	    /* compute the derivatives with n,p */
	    if (pHNode->nodeType != CONTACT) {
	      indexN = pHNode->nEqn;
	      indexP = pHNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pHEdge->dJnDnP1);
	      CMPLX_MULT_SCALAR(prod2, pAc, pHEdge->dJpDpP1);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dy);
	      CMPLX_ADD_ASSIGN(yTotal, prod1);
	    }
	    if (pVNode->nodeType != CONTACT) {
	      indexN = pVNode->nEqn;
	      indexP = pVNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pVEdge->dJnDn);
	      CMPLX_MULT_SCALAR(prod2, pAc, pVEdge->dJpDp);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dx);
	      CMPLX_SUBT_ASSIGN(yTotal, prod1);
	    }
	  }
	  break;
	case 2:
	  /* the BR element */
	  pHNode = pElem->pTRNode;
	  pVNode = pElem->pBLNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  if (pElem->elemType == SEMICON) {
	    /* compute the derivatives with n,p */
	    if (pHNode->nodeType != CONTACT) {
	      indexN = pHNode->nEqn;
	      indexP = pHNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pHEdge->dJnDnP1);
	      CMPLX_MULT_SCALAR(prod2, pAc, pHEdge->dJpDpP1);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dy);
	      CMPLX_ADD_ASSIGN(yTotal, prod1);
	    }
	    if (pVNode->nodeType != CONTACT) {
	      indexN = pVNode->nEqn;
	      indexP = pVNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pVEdge->dJnDnP1);
	      CMPLX_MULT_SCALAR(prod2, pAc, pVEdge->dJpDpP1);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dx);
	      CMPLX_ADD_ASSIGN(yTotal, prod1);
	    }
	  }
	  break;
	case 3:
	  /* the BL element */
	  pHNode = pElem->pTLNode;
	  pVNode = pElem->pBRNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  if (pElem->elemType == SEMICON) {
	    /* compute the derivatives with n,p */
	    if (pHNode->nodeType != CONTACT) {
	      indexN = pHNode->nEqn;
	      indexP = pHNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pHEdge->dJnDn);
	      CMPLX_MULT_SCALAR(prod2, pAc, pHEdge->dJpDp);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dy);
	      CMPLX_SUBT_ASSIGN(yTotal, prod1);
	    }
	    if (pVNode->nodeType != CONTACT) {
	      indexN = pVNode->nEqn;
	      indexP = pVNode->pEqn;
	      CMPLX_ASSIGN_VALUE(nAc, xReal[indexN], xImag[indexN]);
	      CMPLX_ASSIGN_VALUE(pAc, xReal[indexP], xImag[indexP]);
	      CMPLX_MULT_SCALAR(prod1, nAc, pVEdge->dJnDnP1);
	      CMPLX_MULT_SCALAR(prod2, pAc, pVEdge->dJpDpP1);
	      CMPLX_ADD(sum, prod1, prod2);
	      CMPLX_MULT_SCALAR(prod1, sum, 0.5 * pElem->dx);
	      CMPLX_ADD_ASSIGN(yTotal, prod1);
	    }
	  }
	  break;
	}
	if (pElem->elemType == SEMICON) {
	  if (pHNode->nodeType != CONTACT) {
	    indexPsi = pHNode->psiEqn;
	    CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	    temp = 0.5 * pElem->dy * (pHEdge->dJnDpsiP1 + pHEdge->dJpDpsiP1);
	    CMPLX_MULT_SCALAR(prod1, psiAc, temp);
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	    if (delVContact) {
	      CMPLX_ADD_SELF_SCALAR(yTotal, -temp);
	    }
	  }
	  if (pVNode->nodeType != CONTACT) {
	    indexPsi = pVNode->psiEqn;
	    CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	    temp = 0.5 * pElem->dx * (pVEdge->dJnDpsiP1 + pVEdge->dJpDpsiP1);
	    CMPLX_MULT_SCALAR(prod1, psiAc, temp);
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	    if (delVContact) {
	      CMPLX_ADD_SELF_SCALAR(yTotal, -temp);
	    }
	  }
	}
	/* displacement current terms */
	if (pHNode->nodeType != CONTACT) {
	  indexPsi = pHNode->psiEqn;
	  CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	  CMPLX_MULT_SCALAR(prod1, *cOmega, pElem->epsRel * 0.5 * pElem->dyOverDx);
	  CMPLX_MULT(prod2, prod1, psiAc);
	  CMPLX_SUBT_ASSIGN(yTotal, prod2);
	  if (delVContact) {
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	  }
	}
	if (pVNode->nodeType != CONTACT) {
	  indexPsi = pVNode->psiEqn;
	  CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	  CMPLX_MULT_SCALAR(prod1, *cOmega, pElem->epsRel * 0.5 * pElem->dxOverDy);
	  CMPLX_MULT(prod2, prod1, psiAc);
	  CMPLX_SUBT_ASSIGN(yTotal, prod2);
	  if (delVContact) {
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	  }
	}
      }
    }
  }
  return (&yTotal); /* XXX */
}


SPcomplex *
oxideAdmittance(TWOdevice *pDevice, TWOcontact *pContact, BOOLEAN delVContact, 
                double *xReal, double *xImag, SPcomplex *cOmega)
{
  TWOnode *pNode, *pHNode = NULL, *pVNode = NULL;
  TWOedge *pHEdge, *pVEdge;
  int index, i, indexPsi, numContactNodes;
  TWOelem *pElem;
  SPcomplex psiAc;
  SPcomplex prod1, prod2;

  NG_IGNORE(pDevice);

  CMPLX_ASSIGN_VALUE(yTotal, 0.0, 0.0);

  numContactNodes = pContact->numNodes;
  for (index = 0; index < numContactNodes; index++) {
    pNode = pContact->pNodes[index];
    for (i = 0; i <= 3; i++) {
      pElem = pNode->pElems[i];
      if (pElem != NULL) {
	switch (i) {
	case 0:
	  /* the TL element */
	  pHNode = pElem->pBLNode;
	  pVNode = pElem->pTRNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pRightEdge;
	  break;
	case 1:
	  /* the TR element */
	  pHNode = pElem->pBRNode;
	  pVNode = pElem->pTLNode;
	  pHEdge = pElem->pBotEdge;
	  pVEdge = pElem->pLeftEdge;
	  break;
	case 2:
	  /* the BR element */
	  pHNode = pElem->pTRNode;
	  pVNode = pElem->pBLNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pLeftEdge;
	  break;
	case 3:
	  /* the BL element */
	  pHNode = pElem->pTLNode;
	  pVNode = pElem->pBRNode;
	  pHEdge = pElem->pTopEdge;
	  pVEdge = pElem->pRightEdge;
	  break;
	}
	/* displacement current terms */
	if (pHNode->nodeType != CONTACT) {
	  indexPsi = pHNode->psiEqn;
	  CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	  CMPLX_MULT_SCALAR(prod1, *cOmega, pElem->epsRel * 0.5 * pElem->dyOverDx);
	  CMPLX_MULT(prod2, prod1, psiAc);
	  CMPLX_SUBT_ASSIGN(yTotal, prod2);
	  if (delVContact) {
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	  }
	}
	if (pVNode->nodeType != CONTACT) {
	  indexPsi = pVNode->psiEqn;
	  CMPLX_ASSIGN_VALUE(psiAc, xReal[indexPsi], xImag[indexPsi]);
	  CMPLX_MULT_SCALAR(prod1, *cOmega, pElem->epsRel * 0.5 * pElem->dxOverDy);
	  CMPLX_MULT(prod2, prod1, psiAc);
	  CMPLX_SUBT_ASSIGN(yTotal, prod2);
	  if (delVContact) {
	    CMPLX_ADD_ASSIGN(yTotal, prod1);
	  }
	}
      }
    }
  }
  return (&yTotal);
}

void
NUMD2ys(TWOdevice *pDevice, SPcomplex *s, SPcomplex *yIn)
{
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double dxdy;
  double *solnReal, *solnImag;
  double *rhsReal, *rhsImag;
  SPcomplex yAc, *y;
  BOOLEAN deltaVContact = FALSE;
  SPcomplex temp, cOmega;

  /*
   * change context names of solution vectors for ac analysis dcDeltaSolution
   * stores the real part and copiedSolution stores the imaginary part of the
   * ac solution vector
   */
  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  CMPLX_MULT_SCALAR(cOmega, *s, TNorm);
  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  /* solve the system of equations directly */
  if (!OneCarrier) {
    TWO_jacLoad(pDevice);
  } else if (OneCarrier == N_TYPE) {
    TWONjacLoad(pDevice);
  } else if (OneCarrier == P_TYPE) {
    TWOPjacLoad(pDevice);
  }
  storeNewRhs(pDevice, pDevice->pLastContact);

  spSetComplex(pDevice->matrix);
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    if (pElem->elemType == SEMICON) {
      dxdy = 0.25 * pElem->dx * pElem->dy;
      for (index = 0; index <= 3; index++) {
	pNode = pElem->pNodes[index];
	if (pNode->nodeType != CONTACT) {
	  if (!OneCarrier) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  } else if (OneCarrier == N_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	  } else if (OneCarrier == P_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  }
	}
      }
    }
  }

  spFactor(pDevice->matrix);
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
  y = contactAdmittance(pDevice, pDevice->pFirstContact, deltaVContact,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc, y->real, y->imag);
  CMPLX_ASSIGN(*yIn, yAc);
  CMPLX_NEGATE_SELF(*yIn);
  CMPLX_MULT_SELF_SCALAR(*yIn, GNorm * pDevice->width * LNorm);
}


void
NBJT2ys(TWOdevice *pDevice, SPcomplex *s, SPcomplex *yIeVce, SPcomplex *yIcVce,
        SPcomplex *yIeVbe, SPcomplex *yIcVbe)
{
  TWOcontact *pEmitContact = pDevice->pLastContact;
  TWOcontact *pColContact = pDevice->pFirstContact;
  TWOcontact *pBaseContact = pDevice->pFirstContact->next;
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double width = pDevice->width;
  double dxdy;
  double *solnReal, *solnImag;
  double *rhsReal, *rhsImag;
  SPcomplex *y;
  SPcomplex pIeVce, pIcVce, pIeVbe, pIcVbe;
  SPcomplex temp, cOmega;

  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  CMPLX_MULT_SCALAR(cOmega, *s, TNorm);

  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  /* solve the system of equations directly */
  if (!OneCarrier) {
    TWO_jacLoad(pDevice);
  } else if (OneCarrier == N_TYPE) {
    TWONjacLoad(pDevice);
  } else if (OneCarrier == P_TYPE) {
    TWOPjacLoad(pDevice);
  }
  storeNewRhs(pDevice, pColContact);
  spSetComplex(pDevice->matrix);
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    if (pElem->elemType == SEMICON) {
      dxdy = 0.25 * pElem->dx * pElem->dy;
      for (index = 0; index <= 3; index++) {
	pNode = pElem->pNodes[index];
	if (pNode->nodeType != CONTACT) {
	  if (!OneCarrier) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  } else if (OneCarrier == N_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	  } else if (OneCarrier == P_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  }
	}
      }
    }
  }
  spFactor(pDevice->matrix);
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);

  y = contactAdmittance(pDevice, pEmitContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIeVce, y->real, y->imag);
  y = contactAdmittance(pDevice, pColContact, TRUE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIcVce, y->real, y->imag);
  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  storeNewRhs(pDevice, pBaseContact);
  /* don't need to LU factor the jacobian since it exists */
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
  y = contactAdmittance(pDevice, pEmitContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIeVbe, y->real, y->imag);
  y = contactAdmittance(pDevice, pColContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(pIcVbe, y->real, y->imag);


  CMPLX_ASSIGN(*yIeVce, pIeVce);
  CMPLX_ASSIGN(*yIeVbe, pIeVbe);
  CMPLX_ASSIGN(*yIcVce, pIcVce);
  CMPLX_ASSIGN(*yIcVbe, pIcVbe);
  CMPLX_MULT_SELF_SCALAR(*yIeVce, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIeVbe, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIcVce, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(*yIcVbe, GNorm * width * LNorm);
}

void
NUMOSys(TWOdevice *pDevice, SPcomplex *s, struct mosAdmittances *yAc)
{
  TWOcontact *pDContact = pDevice->pFirstContact;
  TWOcontact *pGContact = pDevice->pFirstContact->next;
  TWOcontact *pSContact = pDevice->pFirstContact->next->next;
/*  TWOcontact *pBContact = pDevice->pLastContact; */
  TWOnode *pNode;
  TWOelem *pElem;
  int index, eIndex;
  double width = pDevice->width;
  double dxdy;
  double *rhsReal, *rhsImag;
  double *solnReal, *solnImag;
  SPcomplex *y;
  SPcomplex temp, cOmega;

  pDevice->solverType = SLV_SMSIG;
  rhsReal = pDevice->rhs;
  rhsImag = pDevice->rhsImag;
  solnReal = pDevice->dcDeltaSolution;
  solnImag = pDevice->copiedSolution;

  /* use a normalized radian frequency */
  CMPLX_MULT_SCALAR(cOmega, *s, TNorm);
  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  /* solve the system of equations directly */
  if (!OneCarrier) {
    TWO_jacLoad(pDevice);
  } else if (OneCarrier == N_TYPE) {
    TWONjacLoad(pDevice);
  } else if (OneCarrier == P_TYPE) {
    TWOPjacLoad(pDevice);
  }
  storeNewRhs(pDevice, pDContact);
  spSetComplex(pDevice->matrix);

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    if (pElem->elemType == SEMICON) {
      dxdy = 0.25 * pElem->dx * pElem->dy;
      for (index = 0; index <= 3; index++) {
	pNode = pElem->pNodes[index];
	if (pNode->nodeType != CONTACT) {
	  if (!OneCarrier) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  } else if (OneCarrier == N_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fNN, -temp.real, -temp.imag);
	  } else if (OneCarrier == P_TYPE) {
	    CMPLX_MULT_SCALAR(temp, cOmega, dxdy);
	    spADD_COMPLEX_ELEMENT(pNode->fPP, temp.real, temp.imag);
	  }
	}
      }
    }
  }

  spFactor(pDevice->matrix);
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);

  y = contactAdmittance(pDevice, pDContact, TRUE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIdVdb, y->real, y->imag);
  y = contactAdmittance(pDevice, pSContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIsVdb, y->real, y->imag);
  y = GateTypeAdmittance(pDevice, pGContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIgVdb, y->real, y->imag);

  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  storeNewRhs(pDevice, pSContact);
  /* don't need to LU factor the jacobian since it exists */
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
  y = contactAdmittance(pDevice, pDContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIdVsb, y->real, y->imag);
  y = contactAdmittance(pDevice, pSContact, TRUE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIsVsb, y->real, y->imag);
  y = GateTypeAdmittance(pDevice, pGContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIgVsb, y->real, y->imag);
  for (index = 1; index <= pDevice->numEqns; index++) {
    rhsImag[index] = 0.0;
  }
  storeNewRhs(pDevice, pGContact);
  spSolve(pDevice->matrix, rhsReal, solnReal, rhsImag, solnImag);
  y = contactAdmittance(pDevice, pDContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIdVgb, y->real, y->imag);
  y = contactAdmittance(pDevice, pSContact, FALSE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIsVgb, y->real, y->imag);
  y = GateTypeAdmittance(pDevice, pGContact, TRUE,
      solnReal, solnImag, &cOmega);
  CMPLX_ASSIGN_VALUE(yAc->yIgVgb, y->real, y->imag);

  CMPLX_MULT_SELF_SCALAR(yAc->yIdVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIdVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIdVgb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIsVgb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVdb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVsb, GNorm * width * LNorm);
  CMPLX_MULT_SELF_SCALAR(yAc->yIgVgb, GNorm * width * LNorm);
}
