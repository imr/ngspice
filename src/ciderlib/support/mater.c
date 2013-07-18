/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/material.h"
#include "ngspice/cidersupt.h"


/*
 * Set material info values to their defaults.
 */
void 
MATLdefaults(MaterialInfo *info)
{
  if ((info->material == OXIDE) || (info->material == INSULATOR)) {
    info->type = INSULATOR;
    info->eps = EPS_OX;
    info->affin = AFFIN_OX;
    info->eg0 = EGAP300_OX;
  } else if (info->material == NITRIDE) {
    info->type = INSULATOR;
    info->eps = EPS_NI;
    info->affin = AFFIN_NI;
    info->eg0 = EGAP300_NI;
  } else if (info->material == POLYSILICON) {
    info->type = SEMICON;
    info->eps = EPS_SI;
    info->affin = AFFIN_SI;
    info->nc0 = 0.0;
    info->nv0 = 0.0;
    info->eg0 = EGAP300_SI;
    info->dEgDt = DGAPDT_SI;
    info->trefBGN = TREF_EG_SI;
    info->dEgDn[ELEC] = DGAPDN_N;
    info->dEgDn[HOLE] = DGAPDN_P;
    info->nrefBGN[ELEC] = NBGN_N;
    info->nrefBGN[HOLE] = NBGN_P;
    info->tau0[ELEC] = TAU0_N_SI;
    info->tau0[HOLE] = TAU0_P_SI;
    info->nrefSRH[ELEC] = NSRH_N_SI;
    info->nrefSRH[HOLE] = NSRH_P_SI;
    info->cAug[ELEC] = C_AUG_N_SI;
    info->cAug[HOLE] = C_AUG_P_SI;
    info->aRich[ELEC] = A_RICH_N_SI;
    info->aRich[HOLE] = A_RICH_P_SI;
    info->eDon = E_DON_SI;
    info->eAcc = E_ACC_SI;
    info->gDon = G_DON_SI;
    info->gAcc = G_ACC_SI;
    info->concModel = CT;
    info->muMax[ELEC][MAJOR] = 0.07 * AR_MUMAX_N;
    info->muMin[ELEC][MAJOR] = 0.07 * AR_MUMIN_N;
    info->ntRef[ELEC][MAJOR] = AR_NTREF_N;
    info->ntExp[ELEC][MAJOR] = AR_NTEXP_N;
    info->muMax[HOLE][MAJOR] = 0.07 * AR_MUMAX_P;
    info->muMin[HOLE][MAJOR] = 0.07 * AR_MUMIN_P;
    info->ntRef[HOLE][MAJOR] = AR_NTREF_P;
    info->ntExp[HOLE][MAJOR] = AR_NTEXP_P;
    info->muMax[ELEC][MINOR] = 0.07 * UF_MUMAX_N;
    info->muMin[ELEC][MINOR] = 0.07 * UF_MUMIN_N;
    info->ntRef[ELEC][MINOR] = UF_NTREF_N;
    info->ntExp[ELEC][MINOR] = UF_NTEXP_N;
    info->muMax[HOLE][MINOR] = 0.07 * UF_MUMAX_P;
    info->muMin[HOLE][MINOR] = 0.07 * UF_MUMIN_P;
    info->ntRef[HOLE][MINOR] = UF_NTREF_P;
    info->ntExp[HOLE][MINOR] = UF_NTEXP_P;
    info->fieldModel = CT;
    info->vSat[ELEC] = AR_VSAT_N;
    info->vSat[HOLE] = AR_VSAT_P;
    info->vWarm[ELEC] = SG_VWARM_N;
    info->vWarm[HOLE] = SG_VWARM_P;
    info->mus[ELEC] = 0.07 * MUS_N;
    info->thetaA[ELEC] = THETAA_N;
    info->thetaB[ELEC] = THETAB_N;
    info->mus[HOLE] = 0.07 * MUS_P;
    info->thetaA[HOLE] = THETAA_P;
    info->thetaB[HOLE] = THETAB_P;
  } else if ((info->material == SILICON) || (info->material == SEMICON)) {
    info->type = SEMICON;
    info->eps = EPS_SI;
    info->affin = AFFIN_SI;
    info->nc0 = 0.0;
    info->nv0 = 0.0;
    info->eg0 = EGAP300_SI;
    info->dEgDt = DGAPDT_SI;
    info->trefBGN = TREF_EG_SI;
    info->dEgDn[ELEC] = DGAPDN_N;
    info->dEgDn[HOLE] = DGAPDN_P;
    info->nrefBGN[ELEC] = NBGN_N;
    info->nrefBGN[HOLE] = NBGN_P;
    info->tau0[ELEC] = TAU0_N_SI;
    info->tau0[HOLE] = TAU0_P_SI;
    info->nrefSRH[ELEC] = NSRH_N_SI;
    info->nrefSRH[HOLE] = NSRH_P_SI;
    info->cAug[ELEC] = C_AUG_N_SI;
    info->cAug[HOLE] = C_AUG_P_SI;
    info->aRich[ELEC] = A_RICH_N_SI;
    info->aRich[HOLE] = A_RICH_P_SI;
    info->eDon = E_DON_SI;
    info->eAcc = E_ACC_SI;
    info->gDon = G_DON_SI;
    info->gAcc = G_ACC_SI;
    info->concModel = CT;
    info->muMax[ELEC][MAJOR] = AR_MUMAX_N;
    info->muMin[ELEC][MAJOR] = AR_MUMIN_N;
    info->ntRef[ELEC][MAJOR] = AR_NTREF_N;
    info->ntExp[ELEC][MAJOR] = AR_NTEXP_N;
    info->muMax[HOLE][MAJOR] = AR_MUMAX_P;
    info->muMin[HOLE][MAJOR] = AR_MUMIN_P;
    info->ntRef[HOLE][MAJOR] = AR_NTREF_P;
    info->ntExp[HOLE][MAJOR] = AR_NTEXP_P;
    info->muMax[ELEC][MINOR] = UF_MUMAX_N;
    info->muMin[ELEC][MINOR] = UF_MUMIN_N;
    info->ntRef[ELEC][MINOR] = UF_NTREF_N;
    info->ntExp[ELEC][MINOR] = UF_NTEXP_N;
    info->muMax[HOLE][MINOR] = UF_MUMAX_P;
    info->muMin[HOLE][MINOR] = UF_MUMIN_P;
    info->ntRef[HOLE][MINOR] = UF_NTREF_P;
    info->ntExp[HOLE][MINOR] = UF_NTEXP_P;
    info->fieldModel = CT;
    info->vSat[ELEC] = AR_VSAT_N;
    info->vSat[HOLE] = AR_VSAT_P;
    info->vWarm[ELEC] = SG_VWARM_N;
    info->vWarm[HOLE] = SG_VWARM_P;
    info->mus[ELEC] = MUS_N;
    info->thetaA[ELEC] = THETAA_N;
    info->thetaB[ELEC] = THETAB_N;
    info->mus[HOLE] = MUS_P;
    info->thetaA[HOLE] = THETAA_P;
    info->thetaB[HOLE] = THETAB_P;
  } else if (info->material == GAAS) {
    info->type = SEMICON;
    info->eps = EPS_GA;
    info->affin = AFFIN_GA;
    info->nc0 = NCV_NOM * pow(M_N_GA, 1.5);
    info->nv0 = NCV_NOM * pow(M_P_GA, 1.5);
    info->eg0 = EGAP300_GA;
    info->dEgDt = DGAPDT_GA;
    info->trefBGN = TREF_EG_GA;
    info->dEgDn[ELEC] = DGAPDN_N;
    info->dEgDn[HOLE] = DGAPDN_P;
    info->nrefBGN[ELEC] = NBGN_N;
    info->nrefBGN[HOLE] = NBGN_P;
    info->tau0[ELEC] = TAU0_N_GA;
    info->tau0[HOLE] = TAU0_P_GA;
    info->nrefSRH[ELEC] = NSRH_N_GA;
    info->nrefSRH[HOLE] = NSRH_P_GA;
    info->cAug[ELEC] = C_AUG_N_GA;
    info->cAug[HOLE] = C_AUG_P_GA;
    info->aRich[ELEC] = A_RICH_N_GA;
    info->aRich[HOLE] = A_RICH_P_GA;
    info->eDon = E_DON_GA;
    info->eAcc = E_ACC_GA;
    info->gDon = G_DON_GA;
    info->gAcc = G_ACC_GA;
    info->concModel = GA;
    info->muMax[ELEC][MAJOR] = GA_MUMAX_N;
    info->muMin[ELEC][MAJOR] = GA_MUMIN_N;
    info->ntRef[ELEC][MAJOR] = GA_NTREF_N;
    info->ntExp[ELEC][MAJOR] = GA_NTEXP_N;
    info->muMax[HOLE][MAJOR] = GA_MUMAX_P;
    info->muMin[HOLE][MAJOR] = GA_MUMIN_P;
    info->ntRef[HOLE][MAJOR] = GA_NTREF_P;
    info->ntExp[HOLE][MAJOR] = GA_NTEXP_P;
    info->muMax[ELEC][MINOR] = GA_MUMAX_N;
    info->muMin[ELEC][MINOR] = GA_MUMIN_N;
    info->ntRef[ELEC][MINOR] = GA_NTREF_N;
    info->ntExp[ELEC][MINOR] = GA_NTEXP_N;
    info->muMax[HOLE][MINOR] = GA_MUMAX_P;
    info->muMin[HOLE][MINOR] = GA_MUMIN_P;
    info->ntRef[HOLE][MINOR] = GA_NTREF_P;
    info->ntExp[HOLE][MINOR] = GA_NTEXP_P;
    info->fieldModel = GA;
    info->vSat[ELEC] = GA_VSAT_N;
    info->vSat[HOLE] = GA_VSAT_P;
    info->vWarm[ELEC] = GA_VWARM_N;
    info->vWarm[HOLE] = GA_VWARM_P;
    info->mus[ELEC] = MUS_N;
    info->thetaA[ELEC] = THETAA_N;
    info->thetaB[ELEC] = THETAB_N;
    info->mus[HOLE] = MUS_P;
    info->thetaA[HOLE] = THETAA_P;
    info->thetaB[HOLE] = THETAB_P;
  }
}

/*
 * Compute the temperature-dependent physical parameters of materials
 * Normalize physical constants Actual Instance Temperature is passed in thru
 * the global var 'Temp'
 */
void 
MATLtempDep(MaterialInfo *info, double tnom)
/* double tnom  Nominal Parameter Temperature */
{
  double tmp1;
  double relTemp, perRelTemp;
  double eg0;

  if (info->type == INSULATOR) {
    info->refPsi = RefPsi - (info->affin + 0.5 * info->eg0) / VNorm;
  } else if (info->type == SEMICON) {

    /* compute temperature dependent semiconductor parameters */
    relTemp = Temp / tnom;
    perRelTemp = 1.0 / relTemp;
    tmp1 = pow(relTemp, 1.5);

    /* Bandgap and intrinsic concentration */
    eg0 = info->eg0 + (info->dEgDt * tnom * tnom) / (tnom + info->trefBGN);
    info->eg0 = eg0 - (info->dEgDt * Temp * Temp) / (Temp + info->trefBGN);
    if (info->nc0 > 0.0) {
      info->mass[ELEC] = pow(info->nc0 / NCV_NOM / tmp1, 2.0 / 3.0);
    } else {
      info->mass[ELEC] = 1.039 + 5.477e-4 * Temp - 2.326e-7 * Temp * Temp;
    }
    if (info->nv0 > 0.0) {
      info->mass[HOLE] = pow(info->nv0 / NCV_NOM / tmp1, 2.0 / 3.0);
    } else {
      info->mass[HOLE] = 0.262 * log(0.259 * Temp);
    }
    info->nc0 = NCV_NOM * pow(info->mass[ELEC], 1.5) * tmp1;
    info->nv0 = NCV_NOM * pow(info->mass[HOLE], 1.5) * tmp1;
    info->ni0 = sqrt(info->nc0) * sqrt(info->nv0) *
	exp(-0.5 * info->eg0 / Vt);
    info->refPsi = RefPsi - (info->affin
	+ 0.5 * (info->eg0 + Vt * log(info->nc0 / info->nv0))) / VNorm;

    /* Impurity energies */
    info->eDon /= VNorm;
    info->eAcc /= VNorm;

    /* SRH lifetimes */
    tmp1 = sqrt(perRelTemp) * exp(3.8667 * (perRelTemp - 1.0));
    info->tau0[ELEC] *= tmp1 / TNorm;
    info->tau0[HOLE] *= tmp1 / TNorm;

    /* Auger recombination coefficients */
    info->cAug[ELEC] *= pow(relTemp, 0.14) * NNorm * NNorm * TNorm;
    info->cAug[HOLE] *= pow(relTemp, 0.18) * NNorm * NNorm * TNorm;

    /* Avalanche generation parameters */
    info->aii[ELEC] = AII_N * LNorm;
    info->bii[ELEC] = BII_N / ENorm;
    info->aii[HOLE] = AII_P * LNorm;
    info->bii[HOLE] = BII_P / ENorm;

    /* Effective recombination velocities */
    info->vRich[ELEC] = info->aRich[ELEC] * Temp * Temp /
	(CHARGE * info->nc0 * ENorm);
    info->vRich[HOLE] = info->aRich[HOLE] * Temp * Temp /
	(CHARGE * info->nv0 * ENorm);

    /* Mobility Temperature Dependence */
    MOBtempDep(info, Temp);

    /* Velocity Saturation Parameters */
    info->vSat[ELEC] /= ENorm;
    info->vWarm[ELEC] /= ENorm;
    info->vSat[HOLE] /= ENorm;
    info->vWarm[HOLE] /= ENorm;

    /* Normal Field Mobility Degradation Parameters */
    info->thetaA[ELEC] *= ENorm;
    info->thetaB[ELEC] *= ENorm * ENorm;
    info->thetaA[HOLE] *= ENorm;
    info->thetaB[HOLE] *= ENorm * ENorm;
  }
}

void 
printMaterialInfo(MaterialInfo *info)
{
  static const char tabformat[] = "%12s: % .4e %-12s\t";
  static const char newformat[] = "%12s: % .4e %-12s\n";

  char *name;


  if (info == NULL) {
    fprintf(stderr, "Error: tried to print NIL MaterialInfo\n");
    exit(-1);
  }
  /* Find material name. */
  switch (info->material) {
  case OXIDE:
    name = "OXIDE";
    break;
  case NITRIDE:
    name = "NITRIDE";
    break;
  case INSULATOR:
    name = "INSULATOR";
    break;
  case SILICON:
    name = "SILICON";
    break;
  case POLYSILICON:
    name = "POLYSILICON";
    break;
  case GAAS:
    name = "GAAS";
    break;
  case SEMICON:
    name = "SEMICONDUCTOR";
    break;
  default:
    name = "MATERIAL";
    break;
  }
  if (info->type == INSULATOR) {
    fprintf(stdout, "***** %s PARAMETERS AT %g deg K\n", name, Temp);
    fprintf(stdout, "*** Poisson Equation Parameters -\n");
    fprintf(stdout, tabformat, "Eps", info->eps, "F/cm");
    fprintf(stdout, newformat, "Affin", info->affin, "eV");
    fprintf(stdout, tabformat, "Egap", info->eg0, "eV");
    fprintf(stdout, newformat, "PsiB", -info->refPsi * VNorm, "V");
  } else if (info->type == SEMICON) {
    fprintf(stdout, "***** %s PARAMETERS AT %g deg K\n", name, Temp);
    fprintf(stdout, "*** Poisson Equation\n");
    fprintf(stdout, tabformat, "Eps", info->eps, "F/cm");
    fprintf(stdout, newformat, "Affin", info->affin, "eV");
    fprintf(stdout, tabformat, "Vt", Vt, "V");
    fprintf(stdout, newformat, "Ni", info->ni0, "/cm^3");
    fprintf(stdout, tabformat, "Nc", info->nc0, "/cm^3");
    fprintf(stdout, newformat, "Nv", info->nv0, "/cm^3");
    fprintf(stdout, tabformat, "MnSi", info->mass[ELEC], "*m0 kg");
    fprintf(stdout, newformat, "MpSi", info->mass[HOLE], "*m0 kg");
    fprintf(stdout, tabformat, "Egap", info->eg0, "eV");
    fprintf(stdout, newformat, "PsiB", -info->refPsi * VNorm, "V");
    fprintf(stdout, tabformat, "dEg/dT", info->dEgDt, "eV");
    fprintf(stdout, newformat, "Tref", info->trefBGN, "deg K");
    fprintf(stdout, tabformat, "dEg/dN", info->dEgDn[ELEC], "eV");
    fprintf(stdout, newformat, "Nref", info->nrefBGN[ELEC], "/cm^3");
    fprintf(stdout, tabformat, "dEg/dP", info->dEgDn[HOLE], "eV");
    fprintf(stdout, newformat, "Pref", info->nrefBGN[HOLE], "/cm^3");
    fprintf(stdout, tabformat, "Edon", info->eDon * VNorm, "eV");
    fprintf(stdout, newformat, "Eacc", info->eAcc * VNorm, "eV");
    fprintf(stdout, tabformat, "Gdon", info->gDon, "");
    fprintf(stdout, newformat, "Gacc", info->gAcc, "");
    fprintf(stdout, "*** Generation - Recombination\n");
    fprintf(stdout, tabformat, "Tn0", info->tau0[ELEC] * TNorm, "s");
    fprintf(stdout, newformat, "Tp0", info->tau0[HOLE] * TNorm, "s");
    fprintf(stdout, tabformat, "CnAug",
	info->cAug[ELEC] / (NNorm * NNorm * TNorm), "cm^6/s");
    fprintf(stdout, newformat, "CpAug",
	info->cAug[HOLE] / (NNorm * NNorm * TNorm), "cm^6/s");
    fprintf(stdout, tabformat, "Aiin", info->aii[ELEC] / LNorm, "/cm");
    fprintf(stdout, newformat, "Aiip", info->aii[HOLE] / LNorm, "/cm");
    fprintf(stdout, tabformat, "Biin", info->bii[ELEC] * ENorm, "V/cm");
    fprintf(stdout, newformat, "Biip", info->bii[HOLE] * ENorm, "V/cm");
    fprintf(stdout, "*** Thermionic Emission\n");
    fprintf(stdout, tabformat, "Arichn", info->aRich[ELEC], "A/cm^2/oK^2");
    fprintf(stdout, newformat, "Arichp", info->aRich[HOLE], "A/cm^2/oK^2");
    fprintf(stdout, tabformat, "Vrichn", info->vRich[ELEC] * ENorm, "cm/s");
    fprintf(stdout, newformat, "Vrichp", info->vRich[HOLE] * ENorm, "cm/s");
    fprintf(stdout, "*** Majority Carrier Mobility\n");
    fprintf(stdout, tabformat, "MunMax",
	info->muMax[ELEC][MAJOR], "cm^2/V-s");
    fprintf(stdout, newformat, "MupMax",
	info->muMax[HOLE][MAJOR], "cm^2/V-s");
    fprintf(stdout, tabformat, "MunMin",
	info->muMin[ELEC][MAJOR], "cm^2/V-s");
    fprintf(stdout, newformat, "MupMin",
	info->muMin[HOLE][MAJOR], "cm^2/V-s");
    fprintf(stdout, "*** Minority Carrier Mobility\n");
    fprintf(stdout, tabformat, "MunMax",
	info->muMax[ELEC][MINOR], "cm^2/V-s");
    fprintf(stdout, newformat, "MupMax",
	info->muMax[HOLE][MINOR], "cm^2/V-s");
    fprintf(stdout, tabformat, "MunMin",
	info->muMin[ELEC][MINOR], "cm^2/V-s");
    fprintf(stdout, newformat, "MupMin",
	info->muMin[HOLE][MINOR], "cm^2/V-s");
    fprintf(stdout, "*** Surface Mobility\n");
    fprintf(stdout, tabformat, "Muns", info->mus[ELEC], "cm^2/V-s");
    fprintf(stdout, newformat, "Mups", info->mus[HOLE], "cm^2/V-s");
    fprintf(stdout, tabformat, "ThetaAN", info->thetaA[ELEC] / ENorm, "cm/V");
    fprintf(stdout, newformat, "ThetaAP", info->thetaA[HOLE] / ENorm, "cm/V");
    fprintf(stdout, tabformat, "ThetaBN",
	info->thetaB[ELEC] / ENorm / ENorm, "cm^2/V^2");
    fprintf(stdout, newformat, "ThetaBP",
	info->thetaB[HOLE] / ENorm / ENorm, "cm^2/V^2");
    fprintf(stdout, "*** Velocity Saturation\n");
    fprintf(stdout, tabformat, "VsatN", info->vSat[ELEC] * ENorm, "cm/s");
    fprintf(stdout, newformat, "VsatP", info->vSat[HOLE] * ENorm, "cm/s");
    if (info->fieldModel == SG || info->fieldModel == GA) {
      fprintf(stdout, tabformat, "VwarmN", info->vWarm[ELEC] * ENorm, "cm/s");
      fprintf(stdout, newformat, "VwarmP", info->vWarm[HOLE] * ENorm, "cm/s");
    }
  }
  return;
}
