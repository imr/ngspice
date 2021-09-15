/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "numddefs.h"
#include "ngspice/suffix.h"

IFparm NUMDpTable[] = {		/* parameters */
  IP("off", NUMD_OFF, IF_FLAG, "Initially off"),
  IP("ic.file", NUMD_IC_FILE, IF_STRING, "Initial conditions file"),
  IOP("area", NUMD_AREA, IF_REAL, "Area factor"),
  IP("save", NUMD_PRINT, IF_INTEGER, "Save Solutions"),
  IPR("print", NUMD_PRINT, IF_INTEGER, "Print Solutions"),
  OP("vd", NUMD_VD, IF_REAL, "Voltage"),
  OPR("voltage", NUMD_VD, IF_REAL, "Voltage"),
  OP("id", NUMD_ID, IF_REAL, "Current"),
  OPR("current", NUMD_ID, IF_REAL, "Current"),
  OP("g11", NUMD_G11, IF_REAL, "Conductance"),
  OPR("conductance", NUMD_G11, IF_REAL, "Conductance"),
  OP("c11", NUMD_C11, IF_REAL, "Capacitance"),
  OPR("capacitance", NUMD_C11, IF_REAL, "Capacitance"),
  OP("y11", NUMD_Y11, IF_COMPLEX, "Admittance"),
  OPU("g12", NUMD_G12, IF_REAL, "Conductance"),
  OPU("c12", NUMD_C12, IF_REAL, "Capacitance"),
  OPU("y12", NUMD_Y12, IF_COMPLEX, "Admittance"),
  OPU("g21", NUMD_G21, IF_REAL, "Conductance"),
  OPU("c21", NUMD_C21, IF_REAL, "Capacitance"),
  OPU("y21", NUMD_Y21, IF_COMPLEX, "Admittance"),
  OPU("g22", NUMD_G22, IF_REAL, "Conductance"),
  OPU("c22", NUMD_C22, IF_REAL, "Capacitance"),
  OPU("y22", NUMD_Y22, IF_COMPLEX, "Admittance"),
  IOP("temp", NUMD_TEMP, IF_REAL, "Instance Temperature")
};

IFparm NUMDmPTable[] = {	/* model parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("numd", NUMD_MOD_NUMD, IF_REAL, "Numerical Diode")
};

char *NUMDnames[] = {
  "D+",
  "D-"
};

int NUMDnSize = NUMELEMS(NUMDnames);
int NUMDpTSize = NUMELEMS(NUMDpTable);
int NUMDmPTSize = NUMELEMS(NUMDmPTable);
int NUMDiSize = sizeof(NUMDinstance);
int NUMDmSize = sizeof(NUMDmodel);
