/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

#ifndef NBJT2_H
#define NBJT2_H

/* structures to describe 2d Numerical Bipolar Junction Transistors */

/* circuit level includes */
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"

/* device level includes */
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/profile.h"
#include "ngspice/numglobs.h"
#include "ngspice/carddefs.h"

/* information needed per instance */
typedef struct sNBJT2instance {

  struct GENinstance gen;

#define NBJT2modPtr(inst) ((struct sNBJT2model *)((inst)->gen.GENmodPtr))
#define NBJT2nextInstance(inst) ((struct sNBJT2instance *)((inst)->gen.GENnextInstance))
#define NBJT2name gen.GENname
#define NBJT2state gen.GENstate

#define NBJT2vbe NBJT2state
#define NBJT2vce NBJT2state+1
#define NBJT2ic NBJT2state+2
#define NBJT2ie NBJT2state+3
#define NBJT2dIeDVce NBJT2state+4
#define NBJT2dIeDVbe NBJT2state+5
#define NBJT2dIcDVce NBJT2state+6
#define NBJT2dIcDVbe NBJT2state+7
#define NBJT2numStates 8

  const int NBJT2colNode;		/* number of collector node of bjt */
  const int NBJT2baseNode;		/* number of base node of bjt */
  const int NBJT2emitNode;		/* number of emitter node of bjt */
  double NBJT2width;		/* width factor for the bjt */
  double NBJT2area;		/* area factor for the bjt */
  TWOdevice *NBJT2pDevice;
  GLOBvalues NBJT2globals;	/* Temp.-Dep. Global Parameters */
  int NBJT2type;
  double NBJT2temp;		/* Instance Temperature */
  double NBJT2c11;		/* small-signal capacitance */
  double NBJT2y11r;		/* small-signal admittance, real part */
  double NBJT2y11i;		/* small-signal admittance, imag part */
  double NBJT2c12;		/* small-signal capacitance */
  double NBJT2y12r;		/* small-signal admittance, real part */
  double NBJT2y12i;		/* small-signal admittance, imag part */
  double NBJT2c21;		/* small-signal capacitance */
  double NBJT2y21r;		/* small-signal admittance, real part */
  double NBJT2y21i;		/* small-signal admittance, imag part */
  double NBJT2c22;		/* small-signal capacitance */
  double NBJT2y22r;		/* small-signal admittance, real part */
  double NBJT2y22i;		/* small-signal admittance, imag part */
  int NBJT2print;
  char *NBJT2icFile;            /* Name of initial condition file */
  double *NBJT2colColPtr;	/* pointer to sparse matrix at
				 * (collector,collector) */
  double *NBJT2baseBasePtr;	/* pointer to sparse matrix at (base,base) */
  double *NBJT2emitEmitPtr;	/* pointer to sparse matrix at
				 * (emitter,emitter) */
  double *NBJT2colBasePtr;	/* pointer to sparse matrix at
				 * (collector,base) */
  double *NBJT2colEmitPtr;	/* pointer to sparse matrix at
				 * (collector,emitter) */
  double *NBJT2baseColPtr;	/* pointer to sparse matrix at
				 * (base,collector) */
  double *NBJT2baseEmitPtr;	/* pointer to sparse matrix at (base,emitter) */
  double *NBJT2emitColPtr;	/* pointer to sparse matrix at
				 * (emitter,collector) */
  double *NBJT2emitBasePtr;	/* pointer to sparse matrix at (emitter,base) */
  int NBJT2off;			/* 'off' flag for bjt */
  unsigned NBJT2smSigAvail:1;	/* flag to indicate small-signal done */
  unsigned NBJT2widthGiven:1;	/* flag to indicate width was specified */
  unsigned NBJT2areaGiven:1;	/* flag to indicate area was specified */
  unsigned NBJT2icFileGiven:1;	/* flag to indicate init. cond. file given */
  unsigned NBJT2printGiven:1;	/* flag to indicate print given */
  unsigned NBJT2tempGiven:1;	/* flag to indicate temp given */
} NBJT2instance;

/* per model data */
typedef struct sNBJT2model {	/* model structure for a bjt */

  struct GENmodel gen;

#define NBJT2modType gen.GENmodType
#define NBJT2nextModel(inst) ((struct sNBJT2model *)((inst)->gen.GENnextModel))
#define NBJT2instances(inst) ((NBJT2instance *)((inst)->gen.GENinstances))
#define NBJT2modName gen.GENmodName

  MESHcard *NBJT2xMeshes;	/* list of xmesh cards */
  MESHcard *NBJT2yMeshes;	/* list of ymesh cards */
  DOMNcard *NBJT2domains;	/* list of domain cards */
  BDRYcard *NBJT2boundaries;	/* list of boundary cards */
  DOPcard *NBJT2dopings;	/* list of doping cards */
  ELCTcard *NBJT2electrodes;	/* list of electrode cards */
  CONTcard *NBJT2contacts;	/* list of contact cards */
  MODLcard *NBJT2models;	/* list of model cards */
  MATLcard *NBJT2materials;	/* list of material cards */
  MOBcard *NBJT2mobility;	/* list of mobility cards */
  METHcard *NBJT2methods;	/* list of method cards */
  OPTNcard *NBJT2options;	/* list of option cards */
  OUTPcard *NBJT2outputs;	/* list of output cards */
  TWOtranInfo *NBJT2pInfo;	/* transient analysis information */
  DOPprofile *NBJT2profiles;	/* expanded list of doping profiles */
  DOPtable *NBJT2dopTables;	/* list of tables used by profiles */
  TWOmaterial *NBJT2matlInfo;	/* list of material info structures */
} NBJT2model;

/* type of 2D BJT */
#define NPN 1
#define PNP -1

/* device parameters */
#define NBJT2_WIDTH 1
#define NBJT2_AREA 2
#define NBJT2_OFF 3
#define NBJT2_IC_FILE 4
#define NBJT2_PRINT 7
#define NBJT2_TEMP 8

#define NBJT2_G11 9
#define NBJT2_C11 10
#define NBJT2_Y11 11
#define NBJT2_G12 12
#define NBJT2_C12 13
#define NBJT2_Y12 14
#define NBJT2_G13 15
#define NBJT2_C13 16
#define NBJT2_Y13 17
#define NBJT2_G21 18
#define NBJT2_C21 19
#define NBJT2_Y21 20
#define NBJT2_G22 21
#define NBJT2_C22 22
#define NBJT2_Y22 23
#define NBJT2_G23 24
#define NBJT2_C23 25
#define NBJT2_Y23 26
#define NBJT2_G31 27
#define NBJT2_C31 28
#define NBJT2_Y31 29
#define NBJT2_G32 30
#define NBJT2_C32 31
#define NBJT2_Y32 32
#define NBJT2_G33 33
#define NBJT2_C33 34
#define NBJT2_Y33 35

/* model parameters */
/* NOTE: all true model parameters have been moved to IFcardInfo structures */
#define NBJT2_MOD_NBJT 1

/* device questions */

/* model questions */

#include "nbjt2ext.h"

#endif				/* NBJT2_H */
