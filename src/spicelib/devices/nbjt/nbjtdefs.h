/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

#ifndef NBJT_H
#define NBJT_H

/* data structures used to describe 1D Numerical BJTs */

/* circuit level includes */
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"

/* device level includes */
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/profile.h"
#include "ngspice/numglobs.h"
#include "ngspice/carddefs.h"

/* information needed per instance */
typedef struct sNBJTinstance {

  struct GENinstance gen;

#define NBJTmodPtr(inst) ((struct sNBJTmodel *)((inst)->gen.GENmodPtr))
#define NBJTnextInstance(inst) ((struct sNBJTinstance *)((inst)->gen.GENnextInstance))
#define NBJTname gen.GENname
#define NBJTstate gen.GENstate

#define NBJTvbe NBJTstate
#define NBJTvce NBJTstate+1
#define NBJTic NBJTstate+2
#define NBJTie NBJTstate+3
#define NBJTdIeDVce NBJTstate+4
#define NBJTdIeDVbe NBJTstate+5
#define NBJTdIcDVce NBJTstate+6
#define NBJTdIcDVbe NBJTstate+7
#define NBJTnumStates 8

  const int NBJTcolNode;		/* number of collector node of bjt */
  const int NBJTbaseNode;		/* number of base node of bjt */
  const int NBJTemitNode;		/* number of emitter node of bjt */
  ONEdevice *NBJTpDevice;
  GLOBvalues NBJTglobals;	/* Temp.-Dep. Global Parameters */
  int NBJTtype;
  double NBJTarea;		/* area factor of the BJT */
  double NBJTtemp;		/* Instance Temperature */
  double NBJTc11;		/* small-signal capacitance */
  double NBJTy11r;		/* small-signal admittance, real part */
  double NBJTy11i;		/* small-signal admittance, imag part */
  double NBJTc12;		/* small-signal capacitance */
  double NBJTy12r;		/* small-signal admittance, real part */
  double NBJTy12i;		/* small-signal admittance, imag part */
  double NBJTc21;		/* small-signal capacitance */
  double NBJTy21r;		/* small-signal admittance, real part */
  double NBJTy21i;		/* small-signal admittance, imag part */
  double NBJTc22;		/* small-signal capacitance */
  double NBJTy22r;		/* small-signal admittance, real part */
  double NBJTy22i;		/* small-signal admittance, imag part */
  int NBJTprint;
  char *NBJTicFile;            /* Name of initial condition file */
  double *NBJTcolColPtr;	/* pointer to sparse matrix at
				 * (collector,collector) */
  double *NBJTbaseBasePtr;	/* pointer to sparse matrix at (base,base) */
  double *NBJTemitEmitPtr;	/* pointer to sparse matrix at
				 * (emitter,emitter) */
  double *NBJTcolBasePtr;	/* pointer to sparse matrix at
				 * (collector,base) */
  double *NBJTcolEmitPtr;	/* pointer to sparse matrix at
				 * (collector,emitter) */
  double *NBJTbaseColPtr;	/* pointer to sparse matrix at
				 * (base,collector) */
  double *NBJTbaseEmitPtr;	/* pointer to sparse matrix at (base,emitter) */
  double *NBJTemitColPtr;	/* pointer to sparse matrix at
				 * (emitter,collector) */
  double *NBJTemitBasePtr;	/* pointer to sparse matrix at (emitter,base) */
  int NBJToff;			/* 'off' flag for bjt */
  unsigned NBJTsmSigAvail:1;	/* flag to indicate small-signal done */
  unsigned NBJTareaGiven:1;	/* flag to indicate area was specified */
  unsigned NBJTicFileGiven:1;	/* flag to indicate init. cond. file given */
  unsigned NBJTprintGiven:1;	/* flag to indicate if print was given */
  unsigned NBJTtempGiven:1;	/* flag to indicate if temp was given */
} NBJTinstance;

/* per model data */
typedef struct sNBJTmodel {	/* model structure for a bjt */

  struct GENmodel gen;

#define NBJTmodType gen.GENmodType
#define NBJTnextModel(inst) ((struct sNBJTmodel *)((inst)->gen.GENnextModel))
#define NBJTinstances(inst) ((NBJTinstance *)((inst)->gen.GENinstances))
#define NBJTmodName gen.GENmodName

  MESHcard *NBJTxMeshes;	/* list of xmesh cards */
  MESHcard *NBJTyMeshes;	/* list of ymesh cards */
  DOMNcard *NBJTdomains;	/* list of domain cards */
  BDRYcard *NBJTboundaries;	/* list of boundary cards */
  DOPcard *NBJTdopings;		/* list of doping cards */
  ELCTcard *NBJTelectrodes;	/* list of electrode cards */
  CONTcard *NBJTcontacts;	/* list of contact cards */
  MODLcard *NBJTmodels;		/* list of model cards */
  MATLcard *NBJTmaterials;	/* list of material cards */
  MOBcard *NBJTmobility;	/* list of mobility cards */
  METHcard *NBJTmethods;	/* list of method cards */
  OPTNcard *NBJToptions;	/* list of option cards */
  OUTPcard *NBJToutputs;	/* list of output cards */
  ONEtranInfo *NBJTpInfo;	/* transient analysis information */
  DOPprofile *NBJTprofiles;	/* expanded list of doping profiles */
  DOPtable *NBJTdopTables;	/* list of tables used by profiles */
  ONEmaterial *NBJTmatlInfo;	/* list of material info structures */
} NBJTmodel;

/* type of BJT */
#define NPN 1
#define PNP -1

/* device parameters */
enum {
    NBJT_AREA = 1,
    NBJT_OFF,
    NBJT_IC_FILE,
    NBJT_PRINT,
    NBJT_TEMP,
};

enum {
    NBJT_G11 = 8,
    NBJT_C11,
    NBJT_Y11,
    NBJT_G12,
    NBJT_C12,
    NBJT_Y12,
    NBJT_G13,
    NBJT_C13,
    NBJT_Y13,
    NBJT_G21,
    NBJT_C21,
    NBJT_Y21,
    NBJT_G22,
    NBJT_C22,
    NBJT_Y22,
    NBJT_G23,
    NBJT_C23,
    NBJT_Y23,
    NBJT_G31,
    NBJT_C31,
    NBJT_Y31,
    NBJT_G32,
    NBJT_C32,
    NBJT_Y32,
    NBJT_G33,
    NBJT_C33,
    NBJT_Y33,
};

/* model parameters */
/* NOTE: all true model parameters have been moved to IFcardInfo structures */
#define NBJT_MOD_NBJT 101

/* device questions */

/* model questions */

#include "nbjtext.h"

#endif				/* NBJT_H */
