/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1991 David Gates
**********/
/*
 * Enumerations used by the numerical simulation routines
 */
 
/* Member of CIDER device simulator
 * Version: 1b1
 */


#ifndef ngspice_NUMENUM_H
#define ngspice_NUMENUM_H

#ifdef _MSC_VER
#undef INTERFACE
#endif

/* Doping Profiles */
#define UNIF         101
#define LIN          102
#define GAUSS        103
#define EXP          104
#define ERRFC        105
#define LOOKUP       106
#define X            107
#define Y            108

/* AC-Analysis Method */
#define SOR          201
#define DIRECT       202
#define SOR_ONLY     203

/* One-Carrier-Simulation Types */
#define N_TYPE       301
#define P_TYPE       302

/* Element, Node, and Edge Types */
#define SEMICON      401
#define INSULATOR    402
#define METAL        403
#define INTERFACE    404
#define CONTACT      405
#define SCHOTTKY     406
#define HETERO	     407

/* Material Types */
#define OXIDE        1
#define NITRIDE      2
#define SILICON      3
#define POLYSILICON  4
#define GAAS         5

/* Time-Integration Method */
#ifndef TRAPEZOIDAL
#define TRAPEZOIDAL  1
#define BDF 	     2
#define GEAR	     2
#endif

/* Mobility Models */
#define SG  1			/* Scharfetter-Gummel Model */
#define CT  2			/* Caughey-Thomas Model */
#define AR  3			/* Arora Model */
#define UF  4			/* Univ. of Florida Model */
#define GA  5			/* Gallium-Arsenide Model */
#define TD  6			/* Temperature Dependent */
#define CCS 7			/* Carrier-Carrier Scattering */

/* Carrier Classification */
#define NUM_CARRIERS 2
#define ELEC 0
#define HOLE 1
#define NUM_CARRTYPES 2
#define MAJOR 0
#define MINOR 1

/* Solvers */
#define SLV_NONE     0
#define SLV_EQUIL    1
#define SLV_BIAS     2
#define SLV_SMSIG    3

/* Output Data Formats */
#define RAWFILE 0
#define HDF 1

/* Time and Memory Statistics Types */
#define NUM_STATTYPES	4
#define STAT_SETUP	0
#define STAT_DC		1
#define STAT_TRAN	2
#define STAT_AC		3

#endif
