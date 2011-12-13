/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1990 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/material.h"
#include "twoddefs.h"
#include "twodext.h"

/*
 * These functions calculate the variable-dependence
 * of the surface mobilities
 */

void 
MOBsurfElec(TWOmaterial *info, TWOelem *pElem, double ex, double ey, 
            double es, double wx, double wy, double totalConc)
{
  double thetaA = info->thetaA[ELEC];
  double thetaB = info->thetaB[ELEC];
  double eL, eN, eD, e0, mun;
  double temp1, temp2, temp3, temp4, temp5;
  double temp6, temp7, temp8, temp9;
  double sgnN, sgnL;
  double dMunDEs;                  /* Surface Field Derivative */
  double dMunDEn;                  /* (Local) Normal Field Derivative */
  double dMunDEl;                  /* Tangent Field Derivative */
  double muHC, muSR, muLV;
  double dMuSRDEn;
  double d2MuSRDEn2;
  double dMuHCDEl;
  double dMuHCDMuSR;
  double d2MuHCDMuSR2;
  double d2MuHCDElDMuSR;
  double dEnDEx;                   /* Normal Derivative x Component */
  double dEnDEy;                   /* Normal Derivative y Component */
  double dEnDWx;                   /* Normal Derivative x Component */
  double dEnDWy;                   /* Normal Derivative y Component */
  double dElDEx;                   /* Lateral Derivative x Component */
  double dElDEy;                   /* Lateral Derivative y Component */
  double dElDWx;                   /* Lateral Derivative x Component */
  double dElDWy;                   /* Lateral Derivative y Component */
  
  NG_IGNORE(wx);
  NG_IGNORE(wy);
  NG_IGNORE(totalConc);

  if ( pElem->surface ) { /* replace one field component with surface field */
    if ( pElem->direction == 0 ) {
      ey = es;
    } else {
      ex = es;
    }
  }
  
  e0 = 1.0 / ENorm;
  if ( pElem->direction == 0 ) {
    eN =  ABS( SALPHA_N*ey + SBETA_N*es );
    sgnN = SGN( SALPHA_N*ey + SBETA_N*es );
    eD =  SALPHA_N*( es - ey );
    dEnDEx = 0.0;
    dEnDEy = 1.0;
    dEnDWx = 0.0;
    dEnDWy = 0.0;
    eL =  ABS( ex );
    sgnL = SGN( ex );
    dElDEx = 1.0;
    dElDEy = 0.0;
    dElDWx = 0.0;
    dElDWy = 0.0;
  } else { /* pElem->direction == Y */
    eN =  ABS( SALPHA_N*ex + SBETA_N*es );
    sgnN = SGN( SALPHA_N*ex + SBETA_N*es );
    eD = SALPHA_N*( es - ex );
    dEnDEx = 1.0;
    dEnDEy = 0.0;
    dEnDWx = 0.0;
    dEnDWy = 0.0;
    eL =  ABS( ey );
    sgnL = SGN( ey );
    dElDEx = 0.0;
    dElDEy = 1.0;
    dElDWx = 0.0;
    dElDWy = 0.0;
  }
  /*
    fprintf(stderr,"En = %e, Ep = %e, Ey = %e, Es= %e\n",eN,eL,ey,es);
    */
  
  muLV = pElem->mun0;
  if ( TransDepMobility ) {
    /* Compute various partial derivatives of muSR */
    temp1 = 1.0 / ( 1.0 + thetaA*eN + thetaB*eN*eN );
    temp2 = (thetaA + 2.0*thetaB*eN);
    muSR = muLV * temp1;
    dMuSRDEn = - muSR * temp1 * temp2;
    d2MuSRDEn2 = - 2.0 * (dMuSRDEn * temp1 * temp2 + muSR * temp1 * thetaB);
    if ( FieldDepMobility ) {
      /* Compute various partial derivatives of muHC */
      switch ( info->fieldModel ) {
      case CT:
      case AR:
      case UF:
	temp1 = 1.0 / info->vSat[ELEC];
	temp2 = muSR * temp1;
	temp3 = eL * temp1;
	temp4 = eL * temp2;
	temp5 = 1.0 / ( 1.0 + temp4 * temp4 );
	temp6 = sqrt( temp5 );
	muHC = muSR * temp6;
	dMuHCDMuSR = temp5 * temp6;
	temp7 = temp4 * dMuHCDMuSR;
	temp8 = - 3.0 * temp7 * temp5;
	dMuHCDEl = - muSR * temp7 * temp2;
	d2MuHCDMuSR2 = temp8 * temp3;
	d2MuHCDElDMuSR = temp8 * temp2;
	break;
      case SG:
      default:
	temp1 = 1.0 / info->vSat[ELEC];
	temp2 = muSR * eL * temp1;	/* Vdrift / Vsat */
	temp3 = 1.0 / info->vWarm[ELEC];
	temp4 = muSR * eL * temp3;	/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_N);
	temp6 = 1.0 / (1.0 + temp5*temp4 + temp2*temp2);
	temp7 = sqrt(temp6);
	muHC = muSR * temp7;
	temp7 *= temp6;
	temp8 = (2.0 - temp5)*temp5*temp3 + 2.0*temp2*temp1;
	dMuHCDEl = - 0.5*muSR*temp7*temp8 * muSR;
	temp9 = temp5*temp5;
	dMuHCDMuSR = (1.0 + 0.5*temp9*temp4) * temp7;
	temp9 = (1.5 - temp5)*temp9*temp3 * temp7;
	temp9 -= 1.5 * dMuHCDMuSR * temp6 * temp8;
	d2MuHCDMuSR2 = temp9 * eL;
	d2MuHCDElDMuSR = temp9 * muSR;
	break;
      }
      
      /* Now compute total derivatives */
      temp1 = dMuHCDMuSR * dMuSRDEn * sgnN;
      temp2 = d2MuHCDMuSR2 * dMuSRDEn * dMuSRDEn + dMuHCDMuSR * d2MuSRDEn2;
      temp3 = temp1 - temp2 * eD;
      mun = muHC - temp1 * eD;
      dMunDEn = (temp3 + temp1) * SALPHA_N;
      dMunDEs = temp3 * SBETA_N - temp1 * SALPHA_N;
      dMunDEl = (dMuHCDEl - d2MuHCDElDMuSR * dMuSRDEn * sgnN * eD) * sgnL;
    } else {
      /* Now compute total derivatives */
      temp1 = dMuSRDEn * sgnN;
      temp3 = temp1 - d2MuSRDEn2 * eD;
      mun = muSR - temp1 * eD;
      dMunDEn = (temp3 + temp1) * SALPHA_N;
      dMunDEs = temp3 * SBETA_N - temp1 * SALPHA_N;
      dMunDEl = 0.0;
    }
  } else {
    if ( FieldDepMobility ) {
      /* Compute various partial derivatives of muHC */
      switch ( info->fieldModel ) {
      case CT:
      case AR:
      case UF:
	temp1 = muLV / info->vSat[ELEC];
	temp2 = eL * temp1;
	temp3 = 1.0 / ( 1.0 + temp2 * temp2 );
	temp4 = sqrt( temp3 );
	muHC = muLV * temp4;
	dMuHCDEl = - muHC*temp2*temp3 * temp1;
	break;
      case SG:
      default:
	temp1 = 1.0 / info->vSat[ELEC];
	temp2 = muLV * eL * temp1;	/* Vdrift / Vsat */
	temp3 = 1.0 / info->vWarm[ELEC];
	temp4 = muLV * eL * temp3;	/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_N);
	temp6 = 1.0 / (1.0 + temp5*temp4 + temp2*temp2);
	temp7 = sqrt(temp6);
	muHC = muLV * temp7;
	temp8 = (2.0 - temp5)*temp5*temp3 + 2.0*temp2*temp1;
	dMuHCDEl = - 0.5*muHC*temp6*temp8 * muLV;
	break;
      }
      
      /* Now compute total derivatives */
      mun = muHC;
      dMunDEn = 0.0;
      dMunDEs = 0.0;
      dMunDEl = dMuHCDEl * sgnL;
    } else {
      mun = muLV;
      dMunDEn = 0.0;
      dMunDEs = 0.0;
      dMunDEl = 0.0;
    }
  }

  pElem->mun = mun;
  pElem->dMunDEs = dMunDEs;
  pElem->dMunDEx = dMunDEn * dEnDEx + dMunDEl * dElDEx;
  pElem->dMunDEy = dMunDEn * dEnDEy + dMunDEl * dElDEy;
  pElem->dMunDWx = dMunDEn * dEnDWx + dMunDEl * dElDWx;
  pElem->dMunDWy = dMunDEn * dEnDWy + dMunDEl * dElDWy;

  if ( pElem->surface ) { /* replace one field component with surface field */
    if ( pElem->direction == 0 ) {
      pElem->dMunDEs += pElem->dMunDEy;
      pElem->dMunDEy = 0.0;
    } else {
      pElem->dMunDEs += pElem->dMunDEx;
      pElem->dMunDEx = 0.0;
    }
  }
  
  return;
}

void 
MOBsurfHole(TWOmaterial *info, TWOelem *pElem, double ex, double ey, 
            double es, double wx, double wy, double totalConc)
{
  double thetaA = info->thetaA[HOLE];
  double thetaB = info->thetaB[HOLE];
  double eL, eN, eD, mup;
  double temp1, temp2, temp3, temp4, temp5;
  double temp6, temp7, temp8, temp9;
  double sgnN, sgnL;
  double dMupDEs;                  /* Surface Field Derivative */
  double dMupDEn;                  /* (Local) Normal Field Derivative */
  double dMupDEl;                  /* Tangent Field Derivative */
  double muHC, muSR, muLV;
  double dMuSRDEn;
  double d2MuSRDEn2;
  double dMuHCDEl;
  double dMuHCDMuSR;
  double d2MuHCDMuSR2;
  double d2MuHCDElDMuSR;
  double dEnDEx;                   /* Normal Derivative x Component */
  double dEnDEy;                   /* Normal Derivative y Component */
  double dEnDWx;                   /* Normal Derivative x Component */
  double dEnDWy;                   /* Normal Derivative y Component */
  double dElDEx;                   /* Lateral Derivative x Component */
  double dElDEy;                   /* Lateral Derivative y Component */
  double dElDWx;                   /* Lateral Derivative x Component */
  double dElDWy;                   /* Lateral Derivative y Component */
  
  NG_IGNORE(wx);
  NG_IGNORE(wy);
  NG_IGNORE(totalConc);

  if ( pElem->surface ) { /* replace one field component with surface field */
    if ( pElem->direction == 0 ) {
      ey = es;
    } else {
      ex = es;
    }
  }
  
  if ( pElem->direction == 0 ) {
    eN =  ABS( SALPHA_P*ey + SBETA_P*es );
    sgnN = SGN( SALPHA_P*ey + SBETA_P*es );
    eD = SALPHA_P*( es - ey );
    dEnDEx = 0.0;
    dEnDEy = 1.0;
    dEnDWx = 0.0;
    dEnDWy = 0.0;
    eL =  ABS( ex );
    sgnL = SGN( ex );
    dElDEx = 1.0;
    dElDEy = 0.0;
    dElDWx = 0.0;
    dElDWy = 0.0;
  } else { /* pElem->direction == Y */
    eN =  ABS( SALPHA_P*ex + SBETA_P*es );
    sgnN = SGN( SALPHA_P*ex + SBETA_P*es );
    eD = SALPHA_P*( es - ex );
    dEnDEx = 1.0;
    dEnDEy = 0.0;
    dEnDWx = 0.0;
    dEnDWy = 0.0;
    eL =  ABS( ey );
    sgnL = SGN( ey );
    dElDEx = 0.0;
    dElDEy = 1.0;
    dElDWx = 0.0;
    dElDWy = 0.0;
  }

  muLV = pElem->mup0;
  if ( TransDepMobility ) {
    /* Compute various partial derivatives of muSR */
    temp1 = 1.0 / ( 1.0 + thetaA*eN + thetaB*eN*eN );
    temp2 = thetaA + 2.0*thetaB*eN;
    muSR = muLV * temp1;
    dMuSRDEn = - muSR * temp1 * temp2;
    d2MuSRDEn2 = - 2.0 * (dMuSRDEn * temp1 * temp2 + muSR * temp1 * thetaB);
    if ( FieldDepMobility ) {
      /* Compute various partial derivatives of muHC */
      switch ( info->fieldModel ) {
      case CT:
      case AR:
      case UF:
	temp1 = 1.0 / info->vSat[HOLE];
	temp2 = muSR * temp1;
	temp3 = eL * temp1;
	temp4 = eL * temp2;
	temp5 = 1.0 / ( 1.0 + temp4 );
	muHC = muSR * temp5;
	dMuHCDMuSR = temp5 * temp5;
	dMuHCDEl = - muSR * dMuHCDMuSR * temp2;
	temp6 = - 2.0 * dMuHCDMuSR * temp5;
	d2MuHCDMuSR2 = temp6 * temp3;
	d2MuHCDElDMuSR = temp6 * temp2;
	break;
      case SG:
      default:
	temp1 = 1.0 / info->vSat[HOLE];
	temp2 = muSR * eL * temp1;	/* Vdrift / Vsat */
	temp3 = 1.0 / info->vWarm[HOLE];
	temp4 = muSR * eL * temp3;	/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_P);
	temp6 = 1.0 / (1.0 + temp5*temp4 + temp2*temp2);
	temp7 = sqrt(temp6);
	muHC = muSR * temp7;
	temp7 *= temp6;
	temp8 = (2.0 - temp5)*temp5*temp3 + 2.0*temp2*temp1;
	dMuHCDEl = - 0.5*muSR*temp7*temp8 * muSR;
	temp9 = temp5*temp5;
	dMuHCDMuSR = (1.0 + 0.5*temp9*temp4) * temp7;
	temp9 = (1.5 - temp5)*temp9*temp3 * temp7;
	temp9 -= 1.5 * dMuHCDMuSR * temp6 * temp8;
	d2MuHCDMuSR2 = temp9 * eL;
	d2MuHCDElDMuSR = temp9 * muSR;
	break;
      }
      
      /* Now compute total derivatives */
      temp1 = dMuHCDMuSR * dMuSRDEn * sgnN;
      temp2 = d2MuHCDMuSR2 * dMuSRDEn * dMuSRDEn + dMuHCDMuSR * d2MuSRDEn2;
      temp3 = temp1 - temp2 * eD;
      mup = muHC - temp1 * eD;
      dMupDEn = (temp3 + temp1) * SALPHA_P;
      dMupDEs = temp3 * SBETA_P - temp1 * SALPHA_P;
      dMupDEl = (dMuHCDEl - d2MuHCDElDMuSR * dMuSRDEn * sgnN * eD ) * sgnL;
    } else {
      /* Now compute total derivatives */
      temp1 = dMuSRDEn * sgnN;
      temp3 = temp1 - d2MuSRDEn2 * eD;
      mup = muSR - temp1 * eD;
      dMupDEn = (temp3 + temp1) * SALPHA_P;
      dMupDEs = temp3 * SBETA_P - temp1 * SALPHA_P;
      dMupDEl = 0.0;
    }
  } else {
    if ( FieldDepMobility ) {
      /* Compute various partial derivatives of muHC */
      switch ( info->fieldModel ) {
      case CT:
      case AR:
      case UF:
	temp1 = muLV / info->vSat[HOLE];
	temp2 = eL * temp1;
	temp3 = 1.0 / ( 1.0 + temp2 );
	muHC = muLV * temp3;
	dMuHCDEl = - muHC * temp3 * temp1;
	break;
      case SG:
      default:
	temp1 = 1.0 / info->vSat[HOLE];
	temp2 = muLV * eL * temp1;	/* Vdrift / Vsat */
	temp3 = 1.0 / info->vWarm[HOLE];
	temp4 = muLV * eL * temp3;	/* Vdrift / Vwarm */
	temp5 = temp4 / (temp4 + SG_FIT_P);
	temp6 = 1.0 / (1.0 + temp5*temp4 + temp2*temp2);
	temp7 = sqrt(temp6);
	muHC = muLV * temp7;
	temp8 = (2.0 - temp5)*temp5*temp3 + 2.0*temp2*temp1;
	dMuHCDEl = - 0.5*muHC*temp6*temp8 * muLV;
	break;
      }
      
      /* Now compute total derivatives */
      mup = muHC;
      dMupDEn = 0.0;
      dMupDEs = 0.0;
      dMupDEl = dMuHCDEl * sgnL;
    } else {
      mup = muLV;
      dMupDEn = 0.0;
      dMupDEs = 0.0;
      dMupDEl = 0.0;
    }
  }

  pElem->mup = mup;
  pElem->dMupDEs = dMupDEs;
  pElem->dMupDEx = dMupDEn * dEnDEx + dMupDEl * dElDEx;
  pElem->dMupDEy = dMupDEn * dEnDEy + dMupDEl * dElDEy;
  pElem->dMupDWx = dMupDEn * dEnDWx + dMupDEl * dElDWx;
  pElem->dMupDWy = dMupDEn * dEnDWy + dMupDEl * dElDWy;
  
  if ( pElem->surface ) { /* replace one field component with surface field */
    if ( pElem->direction == 0 ) {
      pElem->dMupDEs += pElem->dMupDEy;
      pElem->dMupDEy = 0.0;
    } else {
      pElem->dMupDEs += pElem->dMupDEx;
      pElem->dMupDEx = 0.0;
    }
  }
  
  return;
}
