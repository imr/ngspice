/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "devdefs.h"
#include "capdefs.h"
#include "suffix.h"

IFparm CAPpTable[] = { /* parameters */ 
 IOPAP("capacitance", CAP_CAP,    IF_REAL, "Device capacitance"),
 IOPAU("ic",          CAP_IC,     IF_REAL, "Initial capacitor voltage"),
 IOPAU("w",           CAP_WIDTH,  IF_REAL, "Device width"),
 IOPAU("l",           CAP_LENGTH, IF_REAL, "Device length"),
 IP("sens_cap",   CAP_CAP_SENS, IF_FLAG, "flag to request sens. WRT cap."),
 OP("i",            CAP_CURRENT,IF_REAL, "Device current"),
 OP("p",            CAP_POWER,  IF_REAL, "Instantaneous device power"),
 OPU("sens_dc",      CAP_QUEST_SENS_DC, IF_REAL,    "dc sensitivity "),
 OPU("sens_real", CAP_QUEST_SENS_REAL, IF_REAL, "real part of ac sensitivity"),
 OPU("sens_imag",CAP_QUEST_SENS_IMAG,IF_REAL,
					"dc sens. & imag part of ac sens."),
 OPU("sens_mag",  CAP_QUEST_SENS_MAG,  IF_REAL, "sensitivity of ac magnitude"),
 OPU("sens_ph",   CAP_QUEST_SENS_PH,   IF_REAL, "sensitivity of ac phase"),
 OPU("sens_cplx", CAP_QUEST_SENS_CPLX, IF_COMPLEX, "ac sensitivity")
};

IFparm CAPmPTable[] = { /* names of model parameters */
 IOPA( "cj",     CAP_MOD_CJ,      IF_REAL, "Bottom Capacitance per area"),
 IOPA( "cjsw",   CAP_MOD_CJSW,    IF_REAL, "Sidewall capacitance per meter"),
 IOPX( "defw",   CAP_MOD_DEFWIDTH,IF_REAL, "Default width"),
 IP( "c",      CAP_MOD_C,       IF_FLAG, "Capacitor model"),
 IOPA( "narrow", CAP_MOD_NARROW,  IF_REAL, "width correction factor")
};

char *CAPnames[] = {
    "C+",
    "C-"
};


int	CAPnSize = NUMELEMS(CAPnames);
int	CAPpTSize = NUMELEMS(CAPpTable);
int	CAPmPTSize = NUMELEMS(CAPmPTable);
int	CAPiSize = sizeof(CAPinstance);
int	CAPmSize = sizeof(CAPmodel);
