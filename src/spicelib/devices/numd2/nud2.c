/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "numd2def.h"
#include "ngspice/suffix.h"

/*
 * This file defines the 2d Numerical Diode data structures that are
 * available to the next level(s) up the calling hierarchy
 */

IFparm NUMD2pTable[] = {	/* parameters */
  IP("off", NUMD2_OFF, IF_FLAG, "Initially off"),
  IP("ic.file", NUMD2_IC_FILE, IF_STRING, "Initial condition file"),
  IOP("w", NUMD2_WIDTH, IF_REAL, "Width factor"),
  IOP("area", NUMD2_AREA, IF_REAL, "Area factor"),
  IP("save", NUMD2_PRINT, IF_INTEGER, "Save solutions"),
  IPR("print", NUMD2_PRINT, IF_INTEGER, "Print solutions"),
  OP("vd", NUMD2_VD, IF_REAL, "Voltage"),
  OPR("voltage", NUMD2_VD, IF_REAL, "Voltage"),
  OP("id", NUMD2_ID, IF_REAL, "Current"),
  OPR("current", NUMD2_ID, IF_REAL, "Current"),
  OP("g11", NUMD2_G11, IF_REAL, "Conductance"),
  OPR("conductance", NUMD2_G11, IF_REAL, "Conductance"),
  OP("c11", NUMD2_C11, IF_REAL, "Capacitance"),
  OPR("capacitance", NUMD2_C11, IF_REAL, "Capacitance"),
  OP("y11", NUMD2_Y11, IF_COMPLEX, "Admittance"),
  OPU("g12", NUMD2_G12, IF_REAL, "Conductance"),
  OPU("c12", NUMD2_C12, IF_REAL, "Capacitance"),
  OPU("y12", NUMD2_Y12, IF_COMPLEX, "Admittance"),
  OPU("g21", NUMD2_G21, IF_REAL, "Conductance"),
  OPU("c21", NUMD2_C21, IF_REAL, "Capacitance"),
  OPU("y21", NUMD2_Y21, IF_COMPLEX, "Admittance"),
  OPU("g22", NUMD2_G22, IF_REAL, "Conductance"),
  OPU("c22", NUMD2_C22, IF_REAL, "Capacitance"),
  OPU("y22", NUMD2_Y22, IF_COMPLEX, "Admittance"),
  IOP("temp", NUMD2_TEMP, IF_REAL, "Instance Temperature")
};

IFparm NUMD2mPTable[] = {	/* model parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("numd", NUMD2_MOD_NUMD, IF_FLAG, "Numerical 2d-Diode Model")
};

char *NUMD2names[] = {
  "Anode",
  "Cathode"
};

int NUMD2nSize = NUMELEMS(NUMD2names);
int NUMD2pTSize = NUMELEMS(NUMD2pTable);
int NUMD2mPTSize = NUMELEMS(NUMD2mPTable);
int NUMD2iSize = sizeof(NUMD2instance);
int NUMD2mSize = sizeof(NUMD2model);
