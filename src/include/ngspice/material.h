/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_MATERIAL_H
#define ngspice_MATERIAL_H

#ifndef NUM_CARRIERS
#include "ngspice/numenum.h"
#endif

/* Information needed by the various physical models */

typedef struct sMaterialInfo
{
  int id;				/* Unique identification number */
  int material;				/* Oxide, Nitride, Silicon, Aluminum */
  int type;				/* Insulator, Semiconductor, Metal */
  double tnom;				/* Parameter Measurement Temp. */

/* Intrinsic-Concentration-Related Parameters */
  double eps;				/* Dielectric Permittivity */
  double affin;				/* Electron Affinity */
  double refPsi;			/* Reference Potential of Intrinsic */
  double ni0;				/* Reference Intrinsic Concentration */
  double nc0;				/* Conduction Band Num States */
  double nv0;				/* Valence Band Num States */
  double mass[NUM_CARRIERS];		/* Conduction Masses */
  double eg0;				/* Band Gap */
  double dEgDt;				/* Temp-Dep Band Gap Narrowing */
  double trefBGN;			/* Ref. Temp for BGN */
  double dEgDn[NUM_CARRIERS];		/* Conc-Dep BGN Constants */
  double nrefBGN[NUM_CARRIERS];		/* Ref. Conc's for BGN */

/* Generation-Recombination Parameters */
  double tau0[NUM_CARRIERS];		/* Low-Conc. SRH Lifetimes */
  double nrefSRH[NUM_CARRIERS];		/* Ref. Conc.'s for Lifetime */
  double cAug[NUM_CARRIERS];		/* Auger Constants */
  double aii[NUM_CARRIERS];		/* Avalanche Factors */
  double bii[NUM_CARRIERS];		/* Aval. Critical Fields */

/* Incomplete Ionization Parameters */
  double eDon;				/* Donor Energy Level */
  double eAcc;				/* Acceptor Energy Level */
  double gDon;				/* Donor Degeneracy Factor */
  double gAcc;				/* Acceptor Degeneracy Factor */

/* Carrier-Velocity Related Parameters */
  double aRich[NUM_CARRIERS];		/* Effective Richardson Constants */
  double vRich[NUM_CARRIERS];		/* Effective Recombination Velocities */

/* Mobility Concentration and Temperature Dependence */
  int concModel;
  int tempModel;
  double muMax[NUM_CARRIERS][NUM_CARRTYPES];
  double muMin[NUM_CARRIERS][NUM_CARRTYPES];
  double ntRef[NUM_CARRIERS][NUM_CARRTYPES];
  double ntExp[NUM_CARRIERS][NUM_CARRTYPES];

/* Mobility Hot Carrier Dependence */
  int fieldModel;
  double vSat[NUM_CARRIERS];
  double vWarm[NUM_CARRIERS];

/* Inversion-Layer Mobility */
  int surfModel;
  double mus[NUM_CARRIERS];
  double thetaA[NUM_CARRIERS];
  double thetaB[NUM_CARRIERS];

  struct sMaterialInfo *next;
} MaterialInfo;
typedef struct sMaterialInfo ONEmaterial;
typedef struct sMaterialInfo TWOmaterial;
typedef struct sMaterialInfo MATLmaterial;

#endif
