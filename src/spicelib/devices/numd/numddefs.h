/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

#ifndef NUMD_H
#define NUMD_H

/* data structures used to describe 1D numerical diodes */

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
typedef struct sNUMDinstance {
  struct sNUMDmodel *NUMDmodPtr;/* back pointer to model */
  struct sNUMDinstance *NUMDnextInstance;	/* pointer to next instance
						 * of current model */
  IFuid NUMDname;		/* pointer to character string naming this
				 * instance */
  int NUMDowner;		/* number of owner process */
  int NUMDstate;		/* pointer to start of state vector for diode */

#define NUMDvoltage NUMDstate
#define NUMDid NUMDstate+1
#define NUMDconduct NUMDstate+2
#define NUMDnumStates 3

  int NUMDposNode;		/* number of positive node of diode */
  int NUMDnegNode;		/* number of negative node of diode */
  ONEdevice *NUMDpDevice;
  GLOBvalues NUMDglobals;	/* Temp.-Dep. Global Parameters */
  int NUMDtype;			/* device type pn or np */
  double NUMDarea;		/* area factor for the diode */
  double NUMDtemp;		/* instance temperature */
  double NUMDc11;		/* small-signal capacitance */
  double NUMDy11r;		/* small-signal admittance, real part */
  double NUMDy11i;		/* small-signal admittance, imag part */
  int NUMDprint;		/* number of timesteps after which print
				 * internal */
  char *NUMDicFile;            /* Name of initial condition file */
  double *NUMDnegPosPtr;	/* pointer to sparse matrix at
				 * (negative,positive) */
  double *NUMDposNegPtr;	/* pointer to sparse matrix at
				 * (positive,negative) */
  double *NUMDposPosPtr;	/* pointer to sparse matrix at
				 * (positive,positive) */
  double *NUMDnegNegPtr;	/* pointer to sparse matrix at
				 * (negative,negative) */

  int NUMDoff;			/* 'off' flag for diode */
  unsigned NUMDsmSigAvail:1;	/* flag to indicate small-signal done */
  unsigned NUMDareaGiven:1;	/* flag to indicate area was specified */
  unsigned NUMDicFileGiven:1;	/* flag to indicate init. cond. file given */
  unsigned NUMDtempGiven:1;	/* flag to indicate temp was specified */
  unsigned NUMDprintGiven:1;	/* flag to indicate if print was specified */
} NUMDinstance;


/* per model data */

typedef struct sNUMDmodel {	/* model structure for a diode */
  int NUMDmodType;		/* type index of this device type */
  struct sNUMDmodel *NUMDnextModel;	/* pointer to next possible model in
					 * linked list */
  NUMDinstance *NUMDinstances;	/* pointer to list of instances that have
				 * this model */
  IFuid NUMDmodName;		/* pointer to character string naming this
				 * model */
  /* Everything below here is numerical-device-specific */
  MESHcard *NUMDxMeshes;	/* list of xmesh cards */
  MESHcard *NUMDyMeshes;	/* list of ymesh cards */
  DOMNcard *NUMDdomains;	/* list of domain cards */
  BDRYcard *NUMDboundaries;	/* list of boundary cards */
  DOPcard *NUMDdopings;		/* list of doping cards */
  ELCTcard *NUMDelectrodes;	/* list of electrode cards */
  CONTcard *NUMDcontacts;	/* list of contact cards */
  MODLcard *NUMDmodels;		/* list of model cards */
  MATLcard *NUMDmaterials;	/* list of material cards */
  MOBcard *NUMDmobility;	/* list of mobility cards */
  METHcard *NUMDmethods;	/* list of method cards */
  OPTNcard *NUMDoptions;	/* list of option cards */
  OUTPcard *NUMDoutputs;	/* list of output cards */
  ONEtranInfo *NUMDpInfo;	/* transient analysis information */
  DOPprofile *NUMDprofiles;	/* expanded list of doping profiles */
  DOPtable *NUMDdopTables;	/* list of tables used by profiles */
  ONEmaterial *NUMDmatlInfo;	/* list of material info structures */
} NUMDmodel;

/* type of 1D diode */
#define PN  1
#define NP -1

/* device parameters */
#define NUMD_AREA 1
#define NUMD_IC_FILE 2
#define NUMD_OFF 3
#define NUMD_PRINT 4
#define NUMD_TEMP 5
#define NUMD_VD 6
#define NUMD_ID 7

#define NUMD_G11 8
#define NUMD_C11 9
#define NUMD_Y11 10
#define NUMD_G12 11
#define NUMD_C12 12
#define NUMD_Y12 13
#define NUMD_G21 14
#define NUMD_C21 15
#define NUMD_Y21 16
#define NUMD_G22 17
#define NUMD_C22 18
#define NUMD_Y22 19

/* model parameters */
/* NOTE: all true model parameters have been moved to IFcardInfo structures */
#define NUMD_MOD_NUMD 101

/* device questions */

/* model questions */

#include "numdext.h"

#endif				/* NUMD_H */
