/**********
Copyright 1992 Regents of the University of California. All rights reserved.
Authors:  1992 David Gates
**********/

#ifndef ngspice_OUTPDEFS_H
#define ngspice_OUTPDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sOUTPcard {
  struct sOUTPcard *OUTPnextCard;
  char *OUTProotFile;
  int OUTPnumVars;
  int OUTPdcDebug;
  int OUTPtranDebug;
  int OUTPacDebug;
  int OUTPgeom;
  int OUTPmesh;
  int OUTPmaterial;
  int OUTPglobals;
  int OUTPstats;
  int OUTPfileType;
  int OUTPdoping;
  int OUTPpsi;
  int OUTPequPsi;
  int OUTPvacPsi;
  int OUTPnConc;
  int OUTPpConc;
  int OUTPphin;
  int OUTPphip;
  int OUTPphic;
  int OUTPphiv;
  int OUTPeField;
  int OUTPjc;
  int OUTPjd;
  int OUTPjn;
  int OUTPjp;
  int OUTPjt;
  int OUTPuNet;
  int OUTPmun;
  int OUTPmup;
  unsigned int OUTProotFileGiven : 1;
  unsigned int OUTPdcDebugGiven : 1;
  unsigned int OUTPtranDebugGiven : 1;
  unsigned int OUTPacDebugGiven : 1;
  unsigned int OUTPgeomGiven : 1;
  unsigned int OUTPmeshGiven : 1;
  unsigned int OUTPmaterialGiven : 1;
  unsigned int OUTPglobalsGiven : 1;
  unsigned int OUTPstatsGiven : 1;
  unsigned int OUTPfileTypeGiven : 1;
  unsigned int OUTPdopingGiven : 1;
  unsigned int OUTPpsiGiven : 1;
  unsigned int OUTPequPsiGiven : 1;
  unsigned int OUTPvacPsiGiven : 1;
  unsigned int OUTPnConcGiven : 1;
  unsigned int OUTPpConcGiven : 1;
  unsigned int OUTPphinGiven : 1;
  unsigned int OUTPphipGiven : 1;
  unsigned int OUTPphicGiven : 1;
  unsigned int OUTPphivGiven : 1;
  unsigned int OUTPeFieldGiven : 1;
  unsigned int OUTPjcGiven : 1;
  unsigned int OUTPjdGiven : 1;
  unsigned int OUTPjnGiven : 1;
  unsigned int OUTPjpGiven : 1;
  unsigned int OUTPjtGiven : 1;
  unsigned int OUTPuNetGiven : 1;
  unsigned int OUTPmunGiven : 1;
  unsigned int OUTPmupGiven : 1;
} OUTPcard;

/* OUTP parameters */
#define OUTP_ALL_DEBUG	1
#define OUTP_DC_DEBUG	2
#define OUTP_TRAN_DEBUG	3
#define OUTP_AC_DEBUG	4
#define OUTP_GEOM    	5
#define OUTP_MESH	6
#define OUTP_MATERIAL	7
#define OUTP_GLOBALS	8
#define OUTP_STATS	9
#define OUTP_ROOTFILE	10
#define OUTP_RAWFILE	11
#define OUTP_HDF	12
#define OUTP_DOPING	13
#define OUTP_PSI	14
#define OUTP_EQU_PSI	15
#define OUTP_VAC_PSI	16
#define OUTP_N_CONC	17
#define OUTP_P_CONC	18
#define OUTP_PHIN	19
#define OUTP_PHIP	20
#define OUTP_PHIC	21
#define OUTP_PHIV	22
#define OUTP_E_FIELD	23
#define OUTP_J_C	24
#define OUTP_J_D	25
#define OUTP_J_N	26
#define OUTP_J_P	27
#define OUTP_J_T	28
#define OUTP_U_NET	29
#define OUTP_MUN	30
#define OUTP_MUP	31

#endif
