/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "numosdef.h"
#include "ngspice/suffix.h"

/*
 * This file defines the 2d Numerical MOSFET data structures that are
 * available to the next level(s) up the calling hierarchy
 */

IFparm NUMOSpTable[] = {	/* parameters */
  IP("off", NUMOS_OFF, IF_FLAG, "Device initially off"),
  IP("ic.file", NUMOS_IC_FILE, IF_STRING, "Initial condition file"),
  IOP("area", NUMOS_AREA, IF_REAL, "Area factor"),
  IOP("w", NUMOS_WIDTH, IF_REAL, "Width factor"),
  IOP("l", NUMOS_LENGTH, IF_REAL, "Length factor"),
  IP("save", NUMOS_PRINT, IF_INTEGER, "Save solutions"),
  IPR("print", NUMOS_PRINT, IF_INTEGER, "Print solutions"),
  OP("g11", NUMOS_G11, IF_REAL, "Conductance"),
  OP("c11", NUMOS_C11, IF_REAL, "Capacitance"),
  OP("y11", NUMOS_Y11, IF_COMPLEX, "Admittance"),
  OP("g12", NUMOS_G12, IF_REAL, "Conductance"),
  OP("c12", NUMOS_C12, IF_REAL, "Capacitance"),
  OP("y12", NUMOS_Y12, IF_COMPLEX, "Admittance"),
  OP("g13", NUMOS_G13, IF_REAL, "Conductance"),
  OP("c13", NUMOS_C13, IF_REAL, "Capacitance"),
  OP("y13", NUMOS_Y13, IF_COMPLEX, "Admittance"),
  OPU("g14", NUMOS_G14, IF_REAL, "Conductance"),
  OPU("c14", NUMOS_C14, IF_REAL, "Capacitance"),
  OPU("y14", NUMOS_Y14, IF_COMPLEX, "Admittance"),
  OP("g21", NUMOS_G21, IF_REAL, "Conductance"),
  OP("c21", NUMOS_C21, IF_REAL, "Capacitance"),
  OP("y21", NUMOS_Y21, IF_COMPLEX, "Admittance"),
  OP("g22", NUMOS_G22, IF_REAL, "Conductance"),
  OP("c22", NUMOS_C22, IF_REAL, "Capacitance"),
  OP("y22", NUMOS_Y22, IF_COMPLEX, "Admittance"),
  OP("g23", NUMOS_G23, IF_REAL, "Conductance"),
  OP("c23", NUMOS_C23, IF_REAL, "Capacitance"),
  OP("y23", NUMOS_Y23, IF_COMPLEX, "Admittance"),
  OPU("g24", NUMOS_G24, IF_REAL, "Conductance"),
  OPU("c24", NUMOS_C24, IF_REAL, "Capacitance"),
  OPU("y24", NUMOS_Y24, IF_COMPLEX, "Admittance"),
  OP("g31", NUMOS_G31, IF_REAL, "Conductance"),
  OP("c31", NUMOS_C31, IF_REAL, "Capacitance"),
  OP("y31", NUMOS_Y31, IF_COMPLEX, "Admittance"),
  OP("g32", NUMOS_G32, IF_REAL, "Conductance"),
  OP("c32", NUMOS_C32, IF_REAL, "Capacitance"),
  OP("y32", NUMOS_Y32, IF_COMPLEX, "Admittance"),
  OP("g33", NUMOS_G33, IF_REAL, "Conductance"),
  OP("c33", NUMOS_C33, IF_REAL, "Capacitance"),
  OP("y33", NUMOS_Y33, IF_COMPLEX, "Admittance"),
  OPU("g34", NUMOS_G34, IF_REAL, "Conductance"),
  OPU("c34", NUMOS_C34, IF_REAL, "Capacitance"),
  OPU("y34", NUMOS_Y34, IF_COMPLEX, "Admittance"),
  OPU("g41", NUMOS_G41, IF_REAL, "Conductance"),
  OPU("c41", NUMOS_C41, IF_REAL, "Capacitance"),
  OPU("y41", NUMOS_Y41, IF_COMPLEX, "Admittance"),
  OPU("g42", NUMOS_G42, IF_REAL, "Conductance"),
  OPU("c42", NUMOS_C42, IF_REAL, "Capacitance"),
  OPU("y42", NUMOS_Y42, IF_COMPLEX, "Admittance"),
  OPU("g43", NUMOS_G43, IF_REAL, "Conductance"),
  OPU("c43", NUMOS_C43, IF_REAL, "Capacitance"),
  OPU("y43", NUMOS_Y43, IF_COMPLEX, "Admittance"),
  OPU("g44", NUMOS_G44, IF_REAL, "Conductance"),
  OPU("c44", NUMOS_C44, IF_REAL, "Capacitance"),
  OPU("y44", NUMOS_Y44, IF_COMPLEX, "Admittance"),
  IOP("temp", NUMOS_TEMP, IF_REAL, "Instance temperature")
};

IFparm NUMOSmPTable[] = {	/* model parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("numos", NUMOS_MOD_NUMOS, IF_FLAG, "Numerical MOSFET"),
};

char *NUMOSnames[] = {
  "Drain",
  "Gate",
  "Source",
  "Substrate"
};

int NUMOSnSize = NUMELEMS(NUMOSnames);
int NUMOSpTSize = NUMELEMS(NUMOSpTable);
int NUMOSmPTSize = NUMELEMS(NUMOSmPTable);
int NUMOSiSize = sizeof(NUMOSinstance);
int NUMOSmSize = sizeof(NUMOSmodel);
