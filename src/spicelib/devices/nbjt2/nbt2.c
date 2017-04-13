/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "nbjt2def.h"
#include "ngspice/suffix.h"

/*
 * This file defines the 2d Numerical BJT data structures that are available
 * to the next level(s) up the calling hierarchy
 */

IFparm NBJT2pTable[] = {	/* parameters */
  IP("off", NBJT2_OFF, IF_FLAG, "Device initially off"),
  IP("ic.file", NBJT2_IC_FILE, IF_STRING, "Initial condition file"),
  IOP("w", NBJT2_WIDTH, IF_REAL, "Width factor"),
  IOP("area", NBJT2_AREA, IF_REAL, "Area factor"),
  IP("save", NBJT2_PRINT, IF_INTEGER, "Save solutions"),
  IPR("print", NBJT2_PRINT, IF_INTEGER, "Print solutions"),
  OP("g11", NBJT2_G11, IF_REAL, "Conductance"),
  OP("c11", NBJT2_C11, IF_REAL, "Capacitance"),
  OP("y11", NBJT2_Y11, IF_COMPLEX, "Admittance"),
  OP("g12", NBJT2_G12, IF_REAL, "Conductance"),
  OP("c12", NBJT2_C12, IF_REAL, "Capacitance"),
  OP("y12", NBJT2_Y12, IF_COMPLEX, "Admittance"),
  OPU("g13", NBJT2_G13, IF_REAL, "Conductance"),
  OPU("c13", NBJT2_C13, IF_REAL, "Capacitance"),
  OPU("y13", NBJT2_Y13, IF_COMPLEX, "Admittance"),
  OP("g21", NBJT2_G21, IF_REAL, "Conductance"),
  OP("c21", NBJT2_C21, IF_REAL, "Capacitance"),
  OP("y21", NBJT2_Y21, IF_COMPLEX, "Admittance"),
  OP("g22", NBJT2_G22, IF_REAL, "Conductance"),
  OP("c22", NBJT2_C22, IF_REAL, "Capacitance"),
  OP("y22", NBJT2_Y22, IF_COMPLEX, "Admittance"),
  OPU("g23", NBJT2_G23, IF_REAL, "Conductance"),
  OPU("c23", NBJT2_C23, IF_REAL, "Capacitance"),
  OPU("y23", NBJT2_Y23, IF_COMPLEX, "Admittance"),
  OPU("g31", NBJT2_G31, IF_REAL, "Conductance"),
  OPU("c31", NBJT2_C31, IF_REAL, "Capacitance"),
  OPU("y31", NBJT2_Y31, IF_COMPLEX, "Admittance"),
  OPU("g32", NBJT2_G32, IF_REAL, "Conductance"),
  OPU("c32", NBJT2_C32, IF_REAL, "Capacitance"),
  OPU("y32", NBJT2_Y32, IF_COMPLEX, "Admittance"),
  OPU("g33", NBJT2_G33, IF_REAL, "Conductance"),
  OPU("c33", NBJT2_C33, IF_REAL, "Capacitance"),
  OPU("y33", NBJT2_Y33, IF_COMPLEX, "Admittance"),
  IOP("temp", NBJT2_TEMP, IF_REAL, "Instance Temperature")
};

IFparm NBJT2mPTable[] = {	/* model parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("nbjt", NBJT2_MOD_NBJT, IF_FLAG, "Numerical BJT Model")
};


char *NBJT2names[] = {
  "Collector",
  "Base",
  "Emitter",
  "Substrate"
};

int NBJT2nSize = NUMELEMS(NBJT2names);
int NBJT2pTSize = NUMELEMS(NBJT2pTable);
int NBJT2mPTSize = NUMELEMS(NBJT2mPTable);
int NBJT2iSize = sizeof(NBJT2instance);
int NBJT2mSize = sizeof(NBJT2model);
