/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/
/*
 * Constants used by the numerical simulation routines
 */
 
 /* Member of CIDER device simulator
 * Version: 1b1
 */
 
#ifndef ngspice_NUMCONST_H
#define ngspice_NUMCONST_H


/* Basic Physical Constants */
#ifdef CHARGE
#undef CHARGE
#endif
#define  CHARGE	                1.602191770e-19		/* C */
#define  BOLTZMANN_CONSTANT	1.38062259e-23		/* J/oK */
#define  VELOCITY_OF_LIGHT	2.997924562e8		/* CM/S */
#define  ELECTRON_MASS		9.10955854e-31		/* kG */
#define  ZERO_DEGREES_CELSIUS	273.15			/* oK */
#define  EPS0		        8.854e-14		/* F/CM */


/* Physical Constants of Silicon, GaAs, SiO2, Si3N4 */
#define EPS_REL_SI 11.7					/* ~EPS0 */
#define EPS_SI     EPS0 * EPS_REL_SI			/* F/CM */
#define EPS_REL_GA 10.9					/* ~EPS0 */
#define EPS_GA     EPS0 * EPS_REL_GA			/* F/CM */
#define EPS_REL_OX 3.9					/* ~EPS0 */
#define EPS_OX     EPS0 * EPS_REL_OX			/* F/CM */
#define EPS_REL_NI 7.5					/* ~EPS0 */
#define EPS_NI     EPS0 * EPS_REL_NI			/* F/CM */

/* Work Function, Affinity, Band & Bandgap Parameters */
#define AFFIN_SI  4.05					/* eV */
#define AFFIN_GA  4.07					/* eV */
#define AFFIN_OX  0.95					/* eV */
#define AFFIN_NI  3.10					/* eV */
#define PHI_METAL  4.10					/* eV */
#define PHI_ALUM   4.10					/* eV */
#define PHI_GOLD   4.75					/* eV */

#define EGAP300_SI 1.1245				/* eV */
#define EGAP300_GA 1.43					/* eV */
#define EGAP300_OX 9.00					/* eV */
#define EGAP300_NI 4.70					/* eV */
#define DGAPDT_SI  4.73e-4				/* eV/oK */
#define DGAPDT_GA  5.405e-4				/* eV/oK */
#define TREF_EG_SI 636.0				/* oK */
#define TREF_EG_GA 204.0				/* oK */

#define NCV_NOM	   2.509e19				/* CM^-3 */
#define M_N_SI     1.447				/* ~ELECTRON_MASS */
#define M_P_SI     1.08					/* ~ELECTRON_MASS */
#define M_N_GA     7.05e-2				/* ~ELECTRON_MASS */
#define M_P_GA     0.427				/* ~ELECTRON_MASS */

/* Physical Model Parameters for Silicon and GaAs*/
/* N = electrons, P = holes */

/* Effective Richardson Constants (ref. PISCES) */
#define A_RICH_N_SI    110.0				/* A/CM^2/oK^2 */
#define A_RICH_P_SI    30.0				/* A/CM^2/oK^2 */
#define A_RICH_N_GA    6.2857				/* A/CM^2/oK^2 */
#define A_RICH_P_GA    105.0				/* A/CM^2/oK^2 */

/* Auger Recombination (ref. PISCES, SOLL90) */
#define C_AUG_N_SI    1.8e-31				/* CM^6/S */
#define C_AUG_P_SI    8.3e-32				/* CM^6/S */
#define C_AUG_N_GA    2.8e-31				/* CM^6/S */
#define C_AUG_P_GA    9.9e-32				/* CM^6/S */

/* SRH Recombination (ref. SOLL90) */
#define TAU0_N_SI     3.0e-5				/* S */
#define NSRH_N_SI     1.0e17				/* CM^-3 */
#define S_N_SI        1.0e4				/* CM/S */
#define TAU0_P_SI     1.0e-5				/* S */
#define NSRH_P_SI     1.0e17				/* CM^-3 */
#define S_P_SI        1.0e4				/* CM/S */
#define TAU0_N_GA     1.0e-7				/* S */
#define NSRH_N_GA     5.0e16				/* CM^-3 */
#define S_N_GA        1.0e4				/* CM/S */
#define TAU0_P_GA     1.0e-7				/* S */
#define NSRH_P_GA     5.0e16				/* CM^-3 */
#define S_P_GA        1.0e4				/* CM/S */

/* Bandgap Narrowing (ref. SOLL90) */
#define DGAPDN_N   1.2e-2				/* V */
#define NBGN_N     1.0e18				/* CM^-3 */
#define DGAPDN_P   9.7e-3				/* V */
#define NBGN_P     1.0e17				/* CM^-3 */

/* Mobility Models : */
/* Scharfetter-Gummel (SG) mobility (ref. SCHA69) */
#define SG_MUMAX_N     1400.0
#define SG_MUMIN_N     75.0
#define SG_NTREF_N     3.0e16
#define SG_NTEXP_N     0.5
#define SG_VSAT_N      1.036e7
#define SG_VWARM_N     4.9e6
#define SG_FIT_N       8.8
#define SG_MUMAX_P     480.0
#define SG_MUMIN_P     53.0
#define SG_NTREF_P     4.0e16
#define SG_NTEXP_P     0.5
#define SG_VSAT_P      1.2e7
#define SG_VWARM_P     2.928e6
#define SG_FIT_P       1.6

/* Caughey-Thomas (CT) mobility (ref. CAUG67) */
#define CT_MUMAX_N     1360.0
#define CT_MUMIN_N     92.0
#define CT_NTREF_N     1.3e17
#define CT_NTEXP_N     0.91
#define CT_VSAT_N      1.1e7
#define CT_MUMAX_P     520.0
#define CT_MUMIN_P     65.0
#define CT_NTREF_P     2.4e17
#define CT_NTEXP_P     0.61
#define CT_VSAT_P      9.5e6

/* Arora (AR) mobility (ref. AROR82) */
#define AR_MUMAX_N     1340.0
#define AR_MUMIN_N     88.0
#define AR_NTREF_N     1.26e17
#define AR_NTEXP_N     0.88
#define AR_VSAT_N      1.38e7
#define AR_MUMAX_P     461.3
#define AR_MUMIN_P     54.3
#define AR_NTREF_P     2.35e17
#define AR_NTEXP_P     0.88
#define AR_VSAT_P      9.0e6

/* Minority Carrier mobility (ref. SOLL90) */
/*
 * These parameters are flawed in that they don't match the majority
 * carrier mobility when the concentration drops to zero.
 * Carrier heating effects must be handled by a different model.
 */
#define UF_MUMAX_N     1412.0
#define UF_MUMIN_N     232.0
#define UF_NTREF_N     8.0e16
#define UF_NTEXP_N     0.9
#define UF_MUMAX_P     500.0
#define UF_MUMIN_P     130.0
#define UF_NTREF_P     8.0e17
#define UF_NTEXP_P     1.25

/* Temperature-Dependence of Arora mobility */
/* Applicable to all above models, but not necessarily accurate. */
#define TD_TREFVS_N    175.0
#define TD_TREFVS_P    312.0
#define TD_EXPMUMAX_N  -2.33
#define TD_EXPMUMAX_P  -2.23
#define TD_EXPMUMIN_N  -0.57
#define TD_EXPMUMIN_P  -0.57
#define TD_EXPNTREF_N  2.4
#define TD_EXPNTREF_P  2.4
#define TD_EXPNTEXP_N  -0.146
#define TD_EXPNTEXP_P  -0.146

/*
 * Inversion-layers are handled differently. They don't fit into the nice
 * pattern established above for bulk mobility.
 */
/* Surface mobility (ref. GATE90) */
#define MUS_N    991.0					/* CM^2/VS */
#define THETAA_N 2.67e-6				/* CM/V */
#define THETAB_N 4.18e-14				/* CM^2/V^2 */
#define SALPHA_N 1.0 / 2.0				/* --- */
#define SBETA_N  1.0 / 2.0				/* --- */
#define MUS_P    240.0					/* CM^2/VS */
#define THETAA_P 3.07e-6				/* CM/V */
#define THETAB_P 0.0					/* CM^2/V^2 */
#define SALPHA_P 2.0 / 3.0				/* --- */
#define SBETA_P  1.0 / 3.0				/* --- */

/* Gallium-Arsenide (GA) mobility (ref. PISCES) */
#define GA_MUMAX_N     5000.0
#define GA_MUMIN_N     50.0
#define GA_NTREF_N     1.0e17
#define GA_NTEXP_N     1.0
#define GA_VSAT_N      7.7e6
#define GA_VWARM_N     2.31e7
#define GA_MUMAX_P     400.0
#define GA_MUMIN_P     40.0
#define GA_NTREF_P     1.0e17
#define GA_NTEXP_P     1.0
#define GA_VSAT_P      7.7e6
#define GA_VWARM_P     2.31e7

/* END OF MOBILITY MODELS */

/* Freeze Out / Incomplete Ionization Parameters */
#define E_ARS_SI  0.049					/* eV (Arsenic) */
#define E_DON_SI  0.044					/* eV (Phosphorus) */
#define E_ACC_SI  0.045					/* eV (Boron) */
#define G_DON_SI  2.0					/* --- */
#define G_ACC_SI  4.0					/* --- */
#define E_DON_GA  0.005					/* eV */
#define E_ACC_GA  0.005					/* eV */
#define G_DON_GA  2.0					/* --- */
#define G_ACC_GA  2.0					/* --- */

/* Impact Ionization / Avalanche Generation Parameters */
/* These are for Silicon.  Need better GaAs parameters. */
#define AII_N  7.03e5
#define BII_N  1.231e6
#define AII_P  1.582e6
#define BII_P  2.036e6

/* Default Surface-State / Fixed-Charge Density */
#define NSS    0.0					/* CM^-2 */

/* Default abstol for Poisson and Current-Continuity Equations */
#define DABSTOL1D   1.0e-12				/* --- */
#define DABSTOL2D   1.0e-8				/* --- */

#endif
