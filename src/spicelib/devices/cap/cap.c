/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "capdefs.h"
#include "ngspice/suffix.h"

IFparm CAPpTable[] = { /* parameters */
    IOPAP("capacitance", CAP_CAP,             IF_REAL, "Device capacitance"),
    IOPAPR("cap",        CAP_CAP,             IF_REAL, "Device capacitance"),
    IOPAPR("c",          CAP_CAP,             IF_REAL, "Device capacitance"),
    IOPAU("ic",          CAP_IC,              IF_REAL, "Initial capacitor voltage"),
    IOPZU("temp",        CAP_TEMP,            IF_REAL, "Instance operating temperature"),
    IOPZ( "dtemp",       CAP_DTEMP,           IF_REAL,
    "Instance temperature difference from the rest of the circuit"),
    IOPAU("w",           CAP_WIDTH,           IF_REAL, "Device width"),
    IOPAU("l",           CAP_LENGTH,          IF_REAL, "Device length"),
    IOPU( "m",           CAP_M,               IF_REAL, "Parallel multiplier"),
    IOPU( "tc1",         CAP_TC1,             IF_REAL, "First order temp. coefficient"),
    IOPU( "tc2",         CAP_TC2,             IF_REAL, "Second order temp. coefficient"),
    IOP(  "bv_max",      CAP_BV_MAX,          IF_REAL, "maximum voltage over capacitance"),
    IOPU( "scale",       CAP_SCALE,           IF_REAL, "Scale factor"),
    IP(   "sens_cap",    CAP_CAP_SENS,        IF_FLAG, "flag to request sens. WRT cap."),
    OP(   "i",           CAP_CURRENT,         IF_REAL, "Device current"),
    OP(   "p",           CAP_POWER,           IF_REAL, "Instantaneous device power"),
    OPU(  "sens_dc",     CAP_QUEST_SENS_DC,   IF_REAL, "dc sensitivity "),
    OPU(  "sens_real",   CAP_QUEST_SENS_REAL, IF_REAL, "real part of ac sensitivity"),
    OPU(  "sens_imag",   CAP_QUEST_SENS_IMAG, IF_REAL,
    "dc sens. & imag part of ac sens."),
    OPU(  "sens_mag",    CAP_QUEST_SENS_MAG,  IF_REAL, "sensitivity of ac magnitude"),
    OPU(  "sens_ph",     CAP_QUEST_SENS_PH,   IF_REAL, "sensitivity of ac phase"),
    OPU(  "sens_cplx",   CAP_QUEST_SENS_CPLX, IF_COMPLEX, "ac sensitivity")
};

IFparm CAPmPTable[] = { /* names of model parameters */
    IOPA( "cap",    CAP_MOD_CAP,      IF_REAL, "Model capacitance"),
    IOPA( "cj",     CAP_MOD_CJ,       IF_REAL, "Bottom Capacitance per area"),
    IOPAR( "cox",   CAP_MOD_CJ,       IF_REAL, "Bottom Capacitance per area"),
    IOPA( "cjsw",   CAP_MOD_CJSW,     IF_REAL, "Sidewall capacitance per meter"),
    IOPAR( "capsw", CAP_MOD_CJSW,     IF_REAL, "Sidewall capacitance per meter"),
    IOPX( "defw",   CAP_MOD_DEFWIDTH, IF_REAL, "Default width"),
    IOPXR( "w",     CAP_MOD_DEFWIDTH, IF_REAL, "Default width"),
    IOPX( "defl",   CAP_MOD_DEFLENGTH,IF_REAL, "Default length"),
    IOPXR( "l",     CAP_MOD_DEFLENGTH,IF_REAL, "Default length"),
    IOPA( "narrow", CAP_MOD_NARROW,   IF_REAL, "width correction factor"),
    IOPA( "short",  CAP_MOD_SHORT,    IF_REAL, "length correction factor"),
    IOPA( "del",    CAP_MOD_DEL,      IF_REAL, "length and width correction factor"),
    IOPA( "tc1",    CAP_MOD_TC1,      IF_REAL, "First order temp. coefficient"),
    IOPA( "tc2",    CAP_MOD_TC2,      IF_REAL, "Second order temp. coefficient"),
    IOPXU("tnom",   CAP_MOD_TNOM,     IF_REAL, "Parameter measurement temperature"),
    IOPA( "di",     CAP_MOD_DI,       IF_REAL, "Relative dielectric constant"),
    IOPA( "thick",  CAP_MOD_THICK,    IF_REAL, "Insulator thickness"),
    IOP(  "bv_max", CAP_MOD_BV_MAX,   IF_REAL, "maximum voltage over capacitance"),
    IP( "c",        CAP_MOD_C,        IF_FLAG, "Capacitor model")
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
