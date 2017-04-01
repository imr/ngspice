/**********
Copyright 1990 Regents of the University of California.  All rights
reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

/*
 * This file defines the LTRA data structures that are available to the next
 * level(s) up the calling hierarchy
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ltradefs.h"
#include "ngspice/suffix.h"

IFparm LTRApTable[] = {		/* parameters */
  IOPAU("v1", LTRA_V1, IF_REAL, "Initial voltage at end 1"),
  IOPAU("v2", LTRA_V2, IF_REAL, "Initial voltage at end 2"),
  IOPAU("i1", LTRA_I1, IF_REAL, "Initial current at end 1"),
  IOPAU("i2", LTRA_I2, IF_REAL, "Initial current at end 2"),
  IP("ic", LTRA_IC, IF_REALVEC, "Initial condition vector:v1,i1,v2,i2"),
  OPU("pos_node1", LTRA_POS_NODE1, IF_INTEGER, "Positive node of end 1 of t-line"),
  OPU("neg_node1", LTRA_NEG_NODE1, IF_INTEGER, "Negative node of end 1 of t.line"),
  OPU("pos_node2", LTRA_POS_NODE2, IF_INTEGER, "Positive node of end 2 of t-line"),
  OPU("neg_node2", LTRA_NEG_NODE2, IF_INTEGER, "Negative node of end 2 of t-line")
};

IFparm LTRAmPTable[] = {	/* model parameters */
  IOP("ltra", LTRA_MOD_LTRA, IF_FLAG, "LTRA model"),
  IOPU("r", LTRA_MOD_R, IF_REAL, "Resistance per metre"),
  IOPAU("l", LTRA_MOD_L, IF_REAL, "Inductance per metre"),
  IOP("g", LTRA_MOD_G, IF_REAL, "Conductance per metre"),
  IOPAU("c", LTRA_MOD_C, IF_REAL, "Capacitance per metre"),
  IOPU("len", LTRA_MOD_LEN, IF_REAL, "length of line"),
  OP("rel", LTRA_MOD_RELTOL, IF_REAL, "Rel. rate of change of deriv. for bkpt"),
  OP("abs", LTRA_MOD_ABSTOL, IF_REAL, "Abs. rate of change of deriv. for bkpt"),

  IOPU("nocontrol", LTRA_MOD_NOCONTROL, IF_FLAG, "No timestep control"),
  IOPU("steplimit", LTRA_MOD_STEPLIMIT, IF_FLAG,
      "always limit timestep to 0.8*(delay of line)"),
  IOPU("nosteplimit", LTRA_MOD_NOSTEPLIMIT, IF_FLAG,
      "don't always limit timestep to 0.8*(delay of line)"),
  IOPU("lininterp", LTRA_MOD_LININTERP, IF_FLAG, "use linear interpolation"),
  IOPU("quadinterp", LTRA_MOD_QUADINTERP, IF_FLAG, "use quadratic interpolation"),
  IOPU("mixedinterp", LTRA_MOD_MIXEDINTERP, IF_FLAG,
      "use linear interpolation if quadratic results look unacceptable"),
  IOPU("truncnr", LTRA_MOD_TRUNCNR, IF_FLAG,
      "use N-R iterations for step calculation in LTRAtrunc"),
  IOPU("truncdontcut", LTRA_MOD_TRUNCDONTCUT, IF_FLAG,
      "don't limit timestep to keep impulse response calculation errors low"),
  IOPAU("compactrel", LTRA_MOD_STLINEREL, IF_REAL,
      "special reltol for straight line checking"),
  IOPAU("compactabs", LTRA_MOD_STLINEABS, IF_REAL,
      "special abstol for straight line checking")
};

char *LTRAnames[] = {
  "P1+",
  "P1-",
  "P2+",
  "P2-"
};

int LTRAnSize = NUMELEMS(LTRAnames);
int LTRApTSize = NUMELEMS(LTRApTable);
int LTRAmPTSize = NUMELEMS(LTRAmPTable);
int LTRAiSize = sizeof(LTRAinstance);
int LTRAmSize = sizeof(LTRAmodel);
