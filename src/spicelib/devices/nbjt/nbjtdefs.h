/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

#ifndef NBJT_H
#define NBJT_H

/* data structures used to describe 1D Numerical BJTs */

/* circuit level includes */
#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"

/* device level includes */
#include "onemesh.h"
#include "onedev.h"
#include "profile.h"
#include "numglobs.h"
#include "carddefs.h"

/* information needed per instance */
typedef struct sNBJTinstance {
  struct sNBJTmodel *NBJTmodPtr;/* back pointer to model */
  struct sNBJTinstance *NBJTnextInstance;	/* pointer to next instance
						 * of current model */
  IFuid NBJTname;		/* pointer to character string naming this
				 * instance */
  int NBJTowner;		/* number of owner process */
  int NBJTstate;		/* pointer to start of state vector for bjt */

  /* entries in the state vector for bjt: */
#define NBJTvbe NBJTstate
#define NBJTvce NBJTstate+1
#define NBJTic NBJTstate+2
#define NBJTie NBJTstate+3
#define NBJTdIeDVce NBJTstate+4
#define NBJTdIeDVbe NBJTstate+5
#define NBJTdIcDVce NBJTstate+6
#define NBJTdIcDVbe NBJTstate+7
#define NBJTnumStates 8

  int NBJTcolNode;		/* number of collector node of bjt */
  int NBJTbaseNode;		/* number of base node of bjt */
  int NBJTemitNode;		/* number of emitter node of bjt */
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
  int NBJTmodType;		/* type index of this device type */
  struct sNBJTmodel *NBJTnextModel;	/* pointer to next possible model in
					 * linked list */
  NBJTinstance *NBJTinstances;	/* pointer to list of instances that have
				 * this model */
  IFuid NBJTmodName;		/* pointer to character string naming this
				 * model */
  /* Everything below here is numerical-device-specific */
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
#define NBJT_AREA 1
#define NBJT_OFF 2
#define NBJT_IC_FILE 3
#define NBJT_PRINT 4
#define NBJT_TEMP 5

#define NBJT_G11 8
#define NBJT_C11 9
#define NBJT_Y11 10
#define NBJT_G12 11
#define NBJT_C12 12
#define NBJT_Y12 13
#define NBJT_G13 14
#define NBJT_C13 15
#define NBJT_Y13 16
#define NBJT_G21 17
#define NBJT_C21 18
#define NBJT_Y21 19
#define NBJT_G22 20
#define NBJT_C22 21
#define NBJT_Y22 22
#define NBJT_G23 23
#define NBJT_C23 24
#define NBJT_Y23 25
#define NBJT_G31 26
#define NBJT_C31 27
#define NBJT_Y31 28
#define NBJT_G32 29
#define NBJT_C32 30
#define NBJT_Y32 31
#define NBJT_G33 32
#define NBJT_C33 33
#define NBJT_Y33 34

/* model parameters */
/* NOTE: all true model parameters have been moved to IFcardInfo structures */
#define NBJT_MOD_NBJT 101

/* device questions */

/* model questions */

#include "nbjtext.h"

#endif				/* NBJT_H */
