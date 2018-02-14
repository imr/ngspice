/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

#ifndef NUMOS_H
#define NUMOS_H

/* data structures used to describe 2D Numerical MOSFETs */

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
typedef struct sNUMOSinstance {

  struct GENinstance gen;

#define NUMOSmodPtr(inst) ((struct sNUMOSmodel *)((inst)->gen.GENmodPtr))
#define NUMOSnextInstance(inst) ((struct sNUMOSinstance *)((inst)->gen.GENnextInstance))
#define NUMOSname gen.GENname
#define NUMOSstate gen.GENstate

#define NUMOSvdb NUMOSstate
#define NUMOSvsb NUMOSstate+1
#define NUMOSvgb NUMOSstate+2
#define NUMOSid  NUMOSstate+3
#define NUMOSis  NUMOSstate+4
#define NUMOSig  NUMOSstate+5
#define NUMOSdIdDVdb NUMOSstate+6
#define NUMOSdIdDVsb NUMOSstate+7
#define NUMOSdIdDVgb NUMOSstate+8
#define NUMOSdIsDVdb NUMOSstate+9
#define NUMOSdIsDVsb NUMOSstate+10
#define NUMOSdIsDVgb NUMOSstate+11
#define NUMOSdIgDVdb NUMOSstate+12
#define NUMOSdIgDVsb NUMOSstate+13
#define NUMOSdIgDVgb NUMOSstate+14
#define NUMOSnumStates 15

  const int NUMOSdrainNode;	/* number of drain node of MOSFET */
  const int NUMOSgateNode;	/* number of gate node of MOSFET */
  const int NUMOSsourceNode;	/* number of source node of MOSFET */
  const int NUMOSbulkNode;	/* number of bulk node of MOSFET */
  double NUMOSarea;		/* area factor for the mosfet */
  double NUMOSwidth;		/* width factor for the mosfet */
  double NUMOSlength;		/* length factor for the mosfet */
  TWOdevice *NUMOSpDevice;
  int NUMOStype;
  double NUMOStemp;		/* Instance Temperature */
  double NUMOSc11;		/* small-signal capacitance */
  double NUMOSy11r;		/* small-signal admittance, real part */
  double NUMOSy11i;		/* small-signal admittance, imag part */
  double NUMOSc12;		/* small-signal capacitance */
  double NUMOSy12r;		/* small-signal admittance, real part */
  double NUMOSy12i;		/* small-signal admittance, imag part */
  double NUMOSc13;		/* small-signal capacitance */
  double NUMOSy13r;		/* small-signal admittance, real part */
  double NUMOSy13i;		/* small-signal admittance, imag part */
  double NUMOSc21;		/* small-signal capacitance */
  double NUMOSy21r;		/* small-signal admittance, real part */
  double NUMOSy21i;		/* small-signal admittance, imag part */
  double NUMOSc22;		/* small-signal capacitance */
  double NUMOSy22r;		/* small-signal admittance, real part */
  double NUMOSy22i;		/* small-signal admittance, imag part */
  double NUMOSc23;		/* small-signal capacitance */
  double NUMOSy23r;		/* small-signal admittance, real part */
  double NUMOSy23i;		/* small-signal admittance, imag part */
  double NUMOSc31;		/* small-signal capacitance */
  double NUMOSy31r;		/* small-signal admittance, real part */
  double NUMOSy31i;		/* small-signal admittance, imag part */
  double NUMOSc32;		/* small-signal capacitance */
  double NUMOSy32r;		/* small-signal admittance, real part */
  double NUMOSy32i;		/* small-signal admittance, imag part */
  double NUMOSc33;		/* small-signal capacitance */
  double NUMOSy33r;		/* small-signal admittance, real part */
  double NUMOSy33i;		/* small-signal admittance, imag part */
  GLOBvalues NUMOSglobals;	/* Temp.-Dep. Global Parameters */
  int NUMOSprint;		/* print timepoints */
  char *NUMOSicFile;            /* Name of initial condition file */
  double *NUMOSdrainDrainPtr;	/* pointer to sparse matrix at (drain,drain) */
  double *NUMOSdrainSourcePtr;	/* pointer to sparse matrix at (drain,source) */
  double *NUMOSdrainGatePtr;	/* pointer to sparse matrix at (drain,gate) */
  double *NUMOSdrainBulkPtr;	/* pointer to sparse matrix at (drain,bulk) */
  double *NUMOSsourceDrainPtr;	/* pointer to sparse matrix at (source,drain) */
  double *NUMOSsourceSourcePtr;	/* pointer to sparse matrix at
				 * (source,source) */
  double *NUMOSsourceGatePtr;	/* pointer to sparse matrix at (source,gate) */
  double *NUMOSsourceBulkPtr;	/* pointer to sparse matrix at (source,bulk) */
  double *NUMOSgateDrainPtr;	/* pointer to sparse matrix at (gate,drain) */
  double *NUMOSgateSourcePtr;	/* pointer to sparse matrix at (gate,source) */
  double *NUMOSgateGatePtr;	/* pointer to sparse matrix at (gate,gate) */
  double *NUMOSgateBulkPtr;	/* pointer to sparse matrix at (gate,bulk) */
  double *NUMOSbulkDrainPtr;	/* pointer to sparse matrix at (bulk,drain) */
  double *NUMOSbulkSourcePtr;	/* pointer to sparse matrix at (bulk,source) */
  double *NUMOSbulkGatePtr;	/* pointer to sparse matrix at (bulk,gate) */
  double *NUMOSbulkBulkPtr;	/* pointer to sparse matrix at (bulk,bulk) */

  int NUMOSoff;			/* 'off' flag for mosfet */
  unsigned NUMOSsmSigAvail:1;	/* flag to indicate small-signal done */
  unsigned NUMOSareaGiven:1;	/* flag to indicate area was specified */
  unsigned NUMOSwidthGiven:1;	/* flag to indicate width was specified */
  unsigned NUMOSlengthGiven:1;	/* flag to indicate length was specified */
  unsigned NUMOSicFileGiven:1;	/* flag to indicate init. cond. file given */
  unsigned NUMOSprintGiven:1;	/* flag to indicate print was given */
  unsigned NUMOStempGiven:1;	/* flag to indicate temp was given */
} NUMOSinstance;

/* per model data */
typedef struct sNUMOSmodel {	/* model structure for a numerical device */

  struct GENmodel gen;

#define NUMOSmodType gen.GENmodType
#define NUMOSnextModel(inst) ((struct sNUMOSmodel *)((inst)->gen.GENnextModel))
#define NUMOSinstances(inst) ((NUMOSinstance *)((inst)->gen.GENinstances))
#define NUMOSmodName gen.GENmodName

  MESHcard *NUMOSxMeshes;	/* list of xmesh cards */
  MESHcard *NUMOSyMeshes;	/* list of ymesh cards */
  DOMNcard *NUMOSdomains;	/* list of domain cards */
  BDRYcard *NUMOSboundaries;	/* list of boundary cards */
  DOPcard *NUMOSdopings;	/* list of doping cards */
  ELCTcard *NUMOSelectrodes;	/* list of electrode cards */
  CONTcard *NUMOScontacts;	/* list of contact cards */
  MODLcard *NUMOSmodels;	/* list of model cards */
  MATLcard *NUMOSmaterials;	/* list of material cards */
  MOBcard *NUMOSmobility;	/* list of mobility cards */
  METHcard *NUMOSmethods;	/* list of method cards */
  OPTNcard *NUMOSoptions;	/* list of option cards */
  OUTPcard *NUMOSoutputs;	/* list of output cards */
  TWOtranInfo *NUMOSpInfo;	/* transient analysis information */
  DOPprofile *NUMOSprofiles;	/* expanded list of doping profiles */
  DOPtable *NUMOSdopTables;	/* list of tables used by profiles */
  TWOmaterial *NUMOSmatlInfo;	/* list of material info structures */
} NUMOSmodel;

/* type of 2D MOSFET */
#define N_CH 1
#define P_CH -1

/* device parameters */
#define NUMOS_AREA 1
#define NUMOS_WIDTH 2
#define NUMOS_LENGTH 3
#define NUMOS_OFF 4
#define NUMOS_IC_FILE 5
#define NUMOS_PRINT 9
#define NUMOS_TEMP 10

#define NUMOS_G11 11
#define NUMOS_C11 12
#define NUMOS_Y11 13
#define NUMOS_G12 14
#define NUMOS_C12 15
#define NUMOS_Y12 16
#define NUMOS_G13 17
#define NUMOS_C13 18
#define NUMOS_Y13 19
#define NUMOS_G14 20
#define NUMOS_C14 21
#define NUMOS_Y14 22
#define NUMOS_G21 23
#define NUMOS_C21 24
#define NUMOS_Y21 25
#define NUMOS_G22 26
#define NUMOS_C22 27
#define NUMOS_Y22 28
#define NUMOS_G23 29
#define NUMOS_C23 30
#define NUMOS_Y23 31
#define NUMOS_G24 32
#define NUMOS_C24 33
#define NUMOS_Y24 34
#define NUMOS_G31 35
#define NUMOS_C31 36
#define NUMOS_Y31 37
#define NUMOS_G32 38
#define NUMOS_C32 39
#define NUMOS_Y32 40
#define NUMOS_G33 41
#define NUMOS_C33 42
#define NUMOS_Y33 43
#define NUMOS_G34 44
#define NUMOS_C34 45
#define NUMOS_Y34 46
#define NUMOS_G41 47
#define NUMOS_C41 48
#define NUMOS_Y41 49
#define NUMOS_G42 50
#define NUMOS_C42 51
#define NUMOS_Y42 52
#define NUMOS_G43 53
#define NUMOS_C43 54
#define NUMOS_Y43 55
#define NUMOS_G44 56
#define NUMOS_C44 57
#define NUMOS_Y44 58

/* model parameters */
/* NOTE: all true model parameters have been moved to IFcardInfo structures */
#define NUMOS_MOD_NUMOS 1

/* device questions */

/* model questions */

#include "numosext.h"

#endif				/* NUMOS_H */
