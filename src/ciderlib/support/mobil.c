/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/macros.h"
#include "ngspice/material.h"
#include "ngspice/cidersupt.h"

void MOBdefaults(MaterialInfo *info , int carrier, int type, 
                 int concmodel, int fieldmodel )
{
  switch (concmodel) {
    case CT: 
      info->concModel = CT;
      if (carrier == ELEC) {
	info->muMax[ELEC][type] = CT_MUMAX_N;
	info->muMin[ELEC][type] = CT_MUMIN_N;
	info->ntRef[ELEC][type] = CT_NTREF_N;
	info->ntExp[ELEC][type] = CT_NTEXP_N;
      } else {
	info->muMax[HOLE][type] = CT_MUMAX_P;
	info->muMin[HOLE][type] = CT_MUMIN_P;
	info->ntRef[HOLE][type] = CT_NTREF_P;
	info->ntExp[HOLE][type] = CT_NTEXP_P;
      }
      break;
    case AR: 
      info->concModel = AR;
      if (carrier == ELEC) {
	info->muMax[ELEC][type] = AR_MUMAX_N;
	info->muMin[ELEC][type] = AR_MUMIN_N;
	info->ntRef[ELEC][type] = AR_NTREF_N;
	info->ntExp[ELEC][type] = AR_NTEXP_N;
      } else {
	info->muMax[HOLE][type] = AR_MUMAX_P;
	info->muMin[HOLE][type] = AR_MUMIN_P;
	info->ntRef[HOLE][type] = AR_NTREF_P;
	info->ntExp[HOLE][type] = AR_NTEXP_P;
      }
      break;
    case UF: 
      info->concModel = UF;
      if (carrier == ELEC) {
	info->muMax[ELEC][type] = UF_MUMAX_N;
	info->muMin[ELEC][type] = UF_MUMIN_N;
	info->ntRef[ELEC][type] = UF_NTREF_N;
	info->ntExp[ELEC][type] = UF_NTEXP_N;
      } else {
	info->muMax[HOLE][type] = UF_MUMAX_P;
	info->muMin[HOLE][type] = UF_MUMIN_P;
	info->ntRef[HOLE][type] = UF_NTREF_P;
	info->ntExp[HOLE][type] = UF_NTEXP_P;
      }
      break;
    case GA:
      info->concModel = GA;
      if (carrier == ELEC) {
	info->muMax[ELEC][type] = GA_MUMAX_N;
	info->muMin[ELEC][type] = GA_MUMIN_N;
	info->ntRef[ELEC][type] = GA_NTREF_N;
	info->ntExp[ELEC][type] = GA_NTEXP_N;
      } else {
	info->muMax[HOLE][type] = GA_MUMAX_P;
	info->muMin[HOLE][type] = GA_MUMIN_P;
	info->ntRef[HOLE][type] = GA_NTREF_P;
	info->ntExp[HOLE][type] = GA_NTEXP_P;
      }
      break;
    case SG:
    default:
      info->concModel = SG;
      if (carrier == ELEC) {
	info->muMax[ELEC][type] = SG_MUMAX_N;
	info->muMin[ELEC][type] = SG_MUMIN_N;
	info->ntRef[ELEC][type] = SG_NTREF_N;
	info->ntExp[ELEC][type] = SG_NTEXP_N;
      } else {
	info->muMax[HOLE][type] = SG_MUMAX_P;
	info->muMin[HOLE][type] = SG_MUMIN_P;
	info->ntRef[HOLE][type] = SG_NTREF_P;
	info->ntExp[HOLE][type] = SG_NTEXP_P;
      }
      break;
  }
  if (type == MAJOR) {
    switch (fieldmodel) {
      case CT: 
	info->fieldModel = CT;
	if (carrier == ELEC) {
	  info->vSat[ELEC] = CT_VSAT_N;
	} else {
	  info->vSat[HOLE] = CT_VSAT_P;
	}
	break;
      case AR: 
      case UF:
	info->fieldModel = AR;
	if (carrier == ELEC) {
	  info->vSat[ELEC] = AR_VSAT_N;
	} else {
	  info->vSat[HOLE] = AR_VSAT_P;
	}
	break;
      case GA:
	info->fieldModel = GA;
	if (carrier == ELEC) {
	  info->vSat[ELEC] = GA_VSAT_N;
	  info->vWarm[ELEC] = GA_VWARM_N;
	} else {
	  info->vSat[HOLE] = GA_VSAT_P;
	  info->vWarm[HOLE] = GA_VWARM_P;
	}
	break;
      case SG:
      default:
	info->fieldModel = SG;
	if (carrier == ELEC) {
	  info->vSat[ELEC] = SG_VSAT_N;
	  info->vWarm[ELEC] = SG_VWARM_N;
	} else {
	  info->vSat[HOLE] = SG_VSAT_P;
	  info->vWarm[HOLE] = SG_VWARM_P;
	}
	break;
    }
  }
}

void
MOBtempDep (MaterialInfo *info, double temp)
{
  double  relTemp = temp / 300.0;
  double  factor, muMin, mu0;

  /* Modify if necessary. */
  if (TempDepMobility)
  {
  /* Do concentration dependence parameters */
    muMin = info->muMin[ELEC][MAJOR];
    mu0 = info->muMax[ELEC][MAJOR] - muMin;
    factor = pow(relTemp, TD_EXPMUMIN_N);
    muMin *= factor;
    factor = pow(relTemp, TD_EXPMUMAX_N);
    mu0 *= factor;
    info->muMin[ELEC][MAJOR] = muMin;
    info->muMax[ELEC][MAJOR] = mu0 + muMin;
    factor = pow(relTemp, TD_EXPNTREF_N);
    info->ntRef[ELEC][MAJOR] *= factor;
    factor = pow(relTemp, TD_EXPNTEXP_N);
    info->ntExp[ELEC][MAJOR] *= factor;

    muMin = info->muMin[ELEC][MINOR];
    mu0 = info->muMax[ELEC][MINOR] - muMin;
    factor = pow(relTemp, TD_EXPMUMIN_N);
    muMin *= factor;
    factor = pow(relTemp, TD_EXPMUMAX_N);
    mu0 *= factor;
    info->muMin[ELEC][MINOR] = muMin;
    info->muMax[ELEC][MINOR] = mu0 + muMin;
    factor = pow(relTemp, TD_EXPNTREF_N);
    info->ntRef[ELEC][MINOR] *= factor;
    factor = pow(relTemp, TD_EXPNTEXP_N);
    info->ntExp[ELEC][MINOR] *= factor;

    muMin = info->muMin[HOLE][MAJOR];
    mu0 = info->muMax[HOLE][MAJOR] - muMin;
    factor = pow(relTemp, TD_EXPMUMIN_P);
    muMin *= factor;
    factor = pow(relTemp, TD_EXPMUMAX_P);
    mu0 *= factor;
    info->muMin[HOLE][MAJOR] = muMin;
    info->muMax[HOLE][MAJOR] = mu0 + muMin;
    factor = pow(relTemp, TD_EXPNTREF_P);
    info->ntRef[HOLE][MAJOR] *= factor;
    factor = pow(relTemp, TD_EXPNTEXP_P);
    info->ntExp[HOLE][MAJOR] *= factor;

    muMin = info->muMin[HOLE][MINOR];
    mu0 = info->muMax[HOLE][MINOR] - muMin;
    factor = pow(relTemp, TD_EXPMUMIN_P);
    muMin *= factor;
    factor = pow(relTemp, TD_EXPMUMAX_P);
    mu0 *= factor;
    info->muMin[HOLE][MINOR] = muMin;
    info->muMax[HOLE][MINOR] = mu0 + muMin;
    factor = pow(relTemp, TD_EXPNTREF_P);
    info->ntRef[HOLE][MINOR] *= factor;
    factor = pow(relTemp, TD_EXPNTEXP_P);
    info->ntExp[HOLE][MINOR] *= factor;

    /* Modify field dependence parameters */
    /* Assume warm carrier reference velocity has same temperature dep. */
    factor = sqrt( tanh( TD_TREFVS_N / Temp ) );
    info->vSat[ELEC] *= factor;
    info->vWarm[ELEC] *= factor;
    factor = sqrt( tanh( TD_TREFVS_P / Temp ) );
    info->vSat[HOLE] *= factor;
    info->vWarm[HOLE] *= factor;
  }
}

void
MOBconcDep (MaterialInfo *info, double conc, double *pMun, double *pMup)
{
  double  s;

  /* We have to check sign of conc even when concentration dependence
   * is not used because it affects whether carriers are majority or
   * minority carriers. Ideally, the minority/majority carrier models
   * should agree at 0.0 concentration, but often they'll be inconsistent.
   */

  if (conc >= 0.0)
  {				/* N type */
    if (ConcDepMobility)
    {
      switch (info->concModel)
      {
	case CT: 
	case AR: 
	case UF: 
	case GA:
	  *pMun = info->muMin[ELEC][MAJOR] +
	    (info->muMax[ELEC][MAJOR] - info->muMin[ELEC][MAJOR]) /
	    (1.0 + pow(conc / info->ntRef[ELEC][MAJOR],
		info->ntExp[ELEC][MAJOR]));

	  *pMup = info->muMin[HOLE][MINOR] +
	    (info->muMax[HOLE][MINOR] - info->muMin[HOLE][MINOR]) /
	    (1.0 + pow(conc / info->ntRef[HOLE][MINOR],
		info->ntExp[HOLE][MINOR]));
	  break;
	case SG: 
	default: 
	  s = info->muMax[ELEC][MAJOR] / info->muMin[ELEC][MAJOR];
	  s = pow(s, 1.0 / info->ntExp[ELEC][MAJOR]) - 1;
	  *pMun = info->muMax[ELEC][MAJOR] /
	    pow(1.0 + conc / (conc / s + info->ntRef[ELEC][MAJOR]),
	      info->ntExp[ELEC][MAJOR]);

	  s = info->muMax[HOLE][MINOR] / info->muMin[HOLE][MINOR];
	  s = pow(s, 1.0 / info->ntExp[HOLE][MINOR]) - 1;
	  *pMup = info->muMax[HOLE][MINOR] /
	    pow(1.0 + conc / (conc / s + info->ntRef[HOLE][MINOR]),
	      info->ntExp[HOLE][MINOR]);
	  break;
      }
    }
    else
    {
      *pMun = info->muMax[ELEC][MAJOR];
      *pMup = info->muMax[HOLE][MINOR];
    }
  }
  else
  {				/* P type */
    if (ConcDepMobility)
    {
      conc = -conc;		/* Take absolute value. */
      switch (info->concModel)
      {
	case CT: 
	case AR: 
	case UF: 
	case GA:
	  *pMun = info->muMin[ELEC][MINOR] +
	    (info->muMax[ELEC][MINOR] - info->muMin[ELEC][MINOR]) /
	    (1.0 + pow(conc / info->ntRef[ELEC][MINOR],
		info->ntExp[ELEC][MINOR]));

	  *pMup = info->muMin[HOLE][MAJOR] +
	    (info->muMax[HOLE][MAJOR] - info->muMin[HOLE][MAJOR]) /
	    (1.0 + pow(conc / info->ntRef[HOLE][MAJOR],
		info->ntExp[HOLE][MAJOR]));
	  break;
	case SG: 
	default: 
	  s = info->muMax[ELEC][MINOR] / info->muMin[ELEC][MINOR];
	  s = pow(s, 1.0 / info->ntExp[ELEC][MINOR]) - 1;
	  *pMun = info->muMax[ELEC][MINOR] /
	    pow(1.0 + conc / (conc / s + info->ntRef[ELEC][MINOR]),
	      info->ntExp[ELEC][MINOR]);

	  s = info->muMax[HOLE][MAJOR] / info->muMin[HOLE][MAJOR];
	  s = pow(s, 1.0 / info->ntExp[HOLE][MAJOR]) - 1;
	  *pMup = info->muMax[HOLE][MAJOR] /
	    pow(1.0 + conc / (conc / s + info->ntRef[HOLE][MAJOR]),
	      info->ntExp[HOLE][MAJOR]);
	  break;
      }
    }
    else
    {
      *pMun = info->muMax[ELEC][MINOR];
      *pMup = info->muMax[HOLE][MAJOR];
    }
  }
  return;
}


void
MOBfieldDep (MaterialInfo *info, int carrier, double field, double *pMu,
             double *pDMu)
{
  double  eLateral, mu;
  double  sgnL;
  double  temp1, temp2, temp3, temp4, temp5, temp6;
  double  dMuDEl;		/* Lateral Field Derivative */

  /* Quick check to make sure we really belong here. */
  if (!FieldDepMobility) /* XXX Global */
    return;

  sgnL = SGN (field);
  eLateral = ABS (field);
  mu = *pMu;			/* Grab temp. and conc.-dep. mobility */

  
  if (carrier == ELEC)
  {
    switch (info->fieldModel)
    {
      case CT: 
      case AR: 
      case UF: 
	temp1 = mu / info->vSat[ELEC];
	temp2 = temp1 * eLateral;
	temp3 = 1.0 / (1.0 + temp2 * temp2);
	mu *= sqrt(temp3);
	dMuDEl = -sgnL * mu * temp3 * temp2 * temp1;
	
	
	break;
      case GA:
	temp1 = info->vSat[ELEC] / info->vWarm[ELEC]; /* Vsat / Vwarm */
	temp2 = mu / info->vWarm[ELEC];
	temp3 = temp2 * eLateral; /* Vdrift / Vwarm */
	temp4 = temp3 * temp3 * temp3;
	temp5 = 1.0  +  temp1 * temp4; 
	temp6 = 1.0 / (1.0 + temp3 * temp4);
	mu *= temp5 * temp6;
	dMuDEl = - sgnL * mu * temp2 *
	    (4.0 * temp4 * temp6 - 3.0 * temp1 * temp3 * temp3 / temp5 ); 
	
	/*
	dMuDEl = 0.0; 
	    */
	break;
      case SG: 
      default: 
	temp1 = mu / info->vSat[ELEC];
	temp2 = temp1 * eLateral;/* Vdrift / Vsat */
	temp3 = mu / info->vWarm[ELEC];
	temp4 = temp3 * eLateral;/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_N);
	temp6 = 1.0 / (1.0 + temp4 * temp5 + temp2 * temp2);
	mu *= sqrt(temp6);
	dMuDEl = -sgnL * 0.5 * mu * temp6 *
	  (temp5 * (2.0 - temp5) * temp3 + (2.0 * temp2 * temp1));
	  
	  
	break;
    }
  }
  else
  {				/* Hole Mobility */
    switch (info->fieldModel)
    {
      case CT: 
      case AR: 
      case UF: 
	temp1 = mu / info->vSat[HOLE];
	temp2 = temp1 * eLateral;
	temp3 = 1.0 / (1.0 + temp2);
	mu *= temp3;
	dMuDEl = -sgnL * mu * temp3 * temp1;
	
	break;
      case GA:
	temp1 = info->vSat[HOLE] / info->vWarm[HOLE]; /* Vsat / Vwarm */
	temp2 = mu / info->vWarm[HOLE];
	temp3 = temp2 * eLateral; /* Vdrift / Vwarm */
	temp4 = temp3 * temp3 * temp3;
	temp5 = 1.0 + temp1 * temp4;
	temp6 = 1.0 / (1.0 + temp3 * temp4);
	mu *= temp5 * temp6;
	dMuDEl = - sgnL * mu * temp2 *
	    (4.0 * temp4 * temp6 - 3.0 * temp1 * temp3 * temp3 / temp5 ); 
	
	
	/*
	dMuDEl = 0.0;
	    */
	break;
      case SG: 
      default: 
	temp1 = mu / info->vSat[HOLE];
	temp2 = temp1 * eLateral;/* Vdrift / Vsat */
	temp3 = mu / info->vWarm[HOLE];
	temp4 = temp3 * eLateral;/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_P);
	temp6 = 1.0 / (1.0 + temp4 * temp5 + temp2 * temp2);
	mu *= sqrt(temp6);
	dMuDEl = -sgnL * 0.5 * mu * temp6 *
	  (temp5 * (2.0 - temp5) * temp3 + (2.0 * temp2 * temp1));
	  
	  
	break;
    }
  }

  *pMu = mu;
  *pDMu = dMuDEl;

  return;
}
