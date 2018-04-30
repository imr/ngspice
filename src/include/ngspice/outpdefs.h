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
enum {
    OUTP_ALL_DEBUG = 1,
    OUTP_DC_DEBUG,
    OUTP_TRAN_DEBUG,
    OUTP_AC_DEBUG,
    OUTP_GEOM,
    OUTP_MESH,
    OUTP_MATERIAL,
    OUTP_GLOBALS,
    OUTP_STATS,
    OUTP_ROOTFILE,
    OUTP_RAWFILE,
    OUTP_HDF,
    OUTP_DOPING,
    OUTP_PSI,
    OUTP_EQU_PSI,
    OUTP_VAC_PSI,
    OUTP_N_CONC,
    OUTP_P_CONC,
    OUTP_PHIN,
    OUTP_PHIP,
    OUTP_PHIC,
    OUTP_PHIV,
    OUTP_E_FIELD,
    OUTP_J_C,
    OUTP_J_D,
    OUTP_J_N,
    OUTP_J_P,
    OUTP_J_T,
    OUTP_U_NET,
    OUTP_MUN,
    OUTP_MUP,
};

#endif
