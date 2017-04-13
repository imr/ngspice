/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "nbjtdefs.h"
#include "ngspice/suffix.h"

/*
 * This file defines the Numerical BJT data structures that are available to
 * the next level(s) up the calling hierarchy
 */

IFparm NBJTpTable[] = {		/* parameters */
  IP("off", NBJT_OFF, IF_FLAG, "Device initially off"),
  IP("ic.file", NBJT_IC_FILE, IF_STRING, "Initial condition file"),
  IOP("area", NBJT_AREA, IF_REAL, "Area factor"),
  IP("save", NBJT_PRINT, IF_INTEGER, "Save Solutions"),
  IPR("print", NBJT_PRINT, IF_INTEGER, "Print Solutions"),
  OP("g11", NBJT_G11, IF_REAL, "Conductance"),
  OP("c11", NBJT_C11, IF_REAL, "Capacitance"),
  OP("y11", NBJT_Y11, IF_COMPLEX, "Admittance"),
  OP("g12", NBJT_G12, IF_REAL, "Conductance"),
  OP("c12", NBJT_C12, IF_REAL, "Capacitance"),
  OP("y12", NBJT_Y12, IF_COMPLEX, "Admittance"),
  OPU("g13", NBJT_G13, IF_REAL, "Conductance"),
  OPU("c13", NBJT_C13, IF_REAL, "Capacitance"),
  OPU("y13", NBJT_Y13, IF_COMPLEX, "Admittance"),
  OP("g21", NBJT_G21, IF_REAL, "Conductance"),
  OP("c21", NBJT_C21, IF_REAL, "Capacitance"),
  OP("y21", NBJT_Y21, IF_COMPLEX, "Admittance"),
  OP("g22", NBJT_G22, IF_REAL, "Conductance"),
  OP("c22", NBJT_C22, IF_REAL, "Capacitance"),
  OP("y22", NBJT_Y22, IF_COMPLEX, "Admittance"),
  OPU("g23", NBJT_G23, IF_REAL, "Conductance"),
  OPU("c23", NBJT_C23, IF_REAL, "Capacitance"),
  OPU("y23", NBJT_Y23, IF_COMPLEX, "Admittance"),
  OPU("g31", NBJT_G31, IF_REAL, "Conductance"),
  OPU("c31", NBJT_C31, IF_REAL, "Capacitance"),
  OPU("y31", NBJT_Y31, IF_COMPLEX, "Admittance"),
  OPU("g32", NBJT_G32, IF_REAL, "Conductance"),
  OPU("c32", NBJT_C32, IF_REAL, "Capacitance"),
  OPU("y32", NBJT_Y32, IF_COMPLEX, "Admittance"),
  OPU("g33", NBJT_G33, IF_REAL, "Conductance"),
  OPU("c33", NBJT_C33, IF_REAL, "Capacitance"),
  OPU("y33", NBJT_Y33, IF_COMPLEX, "Admittance"),
  IOP("temp", NBJT_TEMP, IF_REAL, "Instance Temperature")
};

IFparm NBJTmPTable[] = {	/* model parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("nbjt", NBJT_MOD_NBJT, IF_FLAG, "Numerical BJT Model")
};

char *NBJTnames[] = {
  "Collector",
  "Base",
  "Emitter",
  "Substrate"
};

int NBJTnSize = NUMELEMS(NBJTnames);
int NBJTpTSize = NUMELEMS(NBJTpTable);
int NBJTmPTSize = NUMELEMS(NBJTmPTable);
int NBJTiSize = sizeof(NBJTinstance);
int NBJTmSize = sizeof(NBJTmodel);
