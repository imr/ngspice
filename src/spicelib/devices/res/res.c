/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "resdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"

IFparm RESpTable[] = { /* parameters */
    IOPP(  "resistance", 	RES_RESIST, 	     IF_REAL,    "Resistance"),
    IOPPR( "r", 	RES_RESIST, 	     IF_REAL,    "Resistance"),
    IOPAA( "ac",		RES_ACRESIST,	     IF_REAL,    "AC resistance value"),
    IOPZU( "temp",		RES_TEMP,	     IF_REAL,    "Instance operating temperature"),
    IOPZ(  "dtemp",	RES_DTEMP,	     IF_REAL,
    "Instance temperature difference with the rest of the circuit"),
    IOPQU( "l",		RES_LENGTH,	     IF_REAL,    "Length"),
    IOPZU( "w",		RES_WIDTH,	     IF_REAL,    "Width"),
    IOPU(  "m",		RES_M,		     IF_REAL,    "Multiplication factor"),
    IOPU(  "tc",		RES_TC1,	     IF_REAL,    "First order temp. coefficient"),
    IOPUR( "tc1",		RES_TC1,	     IF_REAL,    "First order temp. coefficient"),
    IOPU(  "tc2",		RES_TC2,	     IF_REAL,    "Second order temp. coefficient"),
    IOPU(  "tce",		RES_TCE,	     IF_REAL,    "exponential temp. coefficient"),
    IOP(   "bv_max",	RES_BV_MAX,	     IF_REAL,    "maximum voltage over resistor"),
    IOPU(  "scale",	RES_SCALE,	     IF_REAL,    "Scale factor"),
    IOP(   "noisy",        RES_NOISY,           IF_INTEGER, "Resistor generate noise"),
    IOPR(  "noise",        RES_NOISY,           IF_INTEGER, "Resistor generate noise"),
    IP(    "sens_resist",  RES_RESIST_SENS,     IF_FLAG,
    "flag to request sensitivity WRT resistance"),
    OP(    "i",            RES_CURRENT,         IF_REAL,    "Current"),
    OP(    "p",            RES_POWER,           IF_REAL,    "Power"),
    OPU(   "sens_dc",      RES_QUEST_SENS_DC,   IF_REAL,    "dc sensitivity "),
    OPU(   "sens_real",    RES_QUEST_SENS_REAL, IF_REAL,
    "dc sensitivity and real part of ac sensitivity"),
    OPU(   "sens_imag",    RES_QUEST_SENS_IMAG, IF_REAL,
    "dc sensitivity and imag part of ac sensitivity"),
    OPU(   "sens_mag",    RES_QUEST_SENS_MAG,   IF_REAL,    "ac sensitivity of magnitude"),
    OPU(   "sens_ph",     RES_QUEST_SENS_PH,    IF_REAL,    "ac sensitivity of phase"),
    OPU(   "sens_cplx",   RES_QUEST_SENS_CPLX,  IF_COMPLEX, "ac sensitivity")
};

IFparm RESmPTable[] = { /* model parameters */
    IOPQ(  "rsh",    RES_MOD_RSH,      IF_REAL,"Sheet resistance"),
    IOPZ(  "narrow", RES_MOD_NARROW,   IF_REAL,"Narrowing of resistor"),
    IOPZR( "dw",     RES_MOD_NARROW,   IF_REAL,"Narrowing of resistor"),
    IOPZ(  "short",  RES_MOD_SHORT,    IF_REAL,"Shortening of resistor"),
    IOPZR( "dlr",    RES_MOD_SHORT,    IF_REAL,"Shortening of resistor"),
    IOPQ(  "tc1",    RES_MOD_TC1,      IF_REAL,"First order temp. coefficient"),
    IOPQR( "tc1r",   RES_MOD_TC1,      IF_REAL,"First order temp. coefficient"),
    IOPQO( "tc2",    RES_MOD_TC2,      IF_REAL,"Second order temp. coefficient"),
    IOPQOR("tc2r",   RES_MOD_TC2,      IF_REAL,"Second order temp. coefficient"),
    IOPQ(  "tce",    RES_MOD_TCE,      IF_REAL,"exponential temp. coefficient"),
    IOPX(  "defw",   RES_MOD_DEFWIDTH, IF_REAL,"Default device width"),
    IOPXR(  "w",     RES_MOD_DEFWIDTH, IF_REAL,"Default device width"),
    IOPX(  "l",      RES_MOD_DEFLENGTH,IF_REAL,"Default device length"),
    IOPQ(  "kf",     RES_MOD_KF,       IF_REAL,"Flicker noise coefficient"),
    IOPQ(  "af",     RES_MOD_AF,       IF_REAL,"Flicker noise exponent"),
    IOPXU( "tnom",   RES_MOD_TNOM,     IF_REAL,"Parameter measurement temperature"),
    IOP(   "r",      RES_MOD_R,        IF_REAL,"Resistor model default value"),
    IOPR(  "res",    RES_MOD_R,        IF_REAL,"Resistor model default value"),
    IOP(   "bv_max", RES_MOD_BV_MAX,   IF_REAL,"maximum voltage over resistor"),
    IOP(   "lf",     RES_MOD_LF,       IF_REAL,"noise area length exponent"),
    IOP(   "wf",     RES_MOD_WF,       IF_REAL,"noise area width exponent"),
    IOP(   "ef",     RES_MOD_EF,       IF_REAL,"noise frequency exponent")
};

char *RESnames[] = {
    "R+",
    "R-"
};

int	RESnSize = NUMELEMS(RESnames);
int	RESpTSize = NUMELEMS(RESpTable);
int	RESmPTSize = NUMELEMS(RESmPTable);
int	RESiSize = sizeof(RESinstance);
int	RESmSize = sizeof(RESmodel);
