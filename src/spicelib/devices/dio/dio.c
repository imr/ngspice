/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "devdefs.h"
#include "diodefs.h"
#include "suffix.h"

IFparm DIOpTable[] = { /* parameters */ 
 IOPU("off",    DIO_OFF,    IF_FLAG, "Initially off"),
 IOPU("temp",   DIO_TEMP,   IF_REAL, "Instance temperature"),
 IOPAU("ic",     DIO_IC,     IF_REAL, "Initial device voltage"),
 IOPU("area",   DIO_AREA,   IF_REAL, "Area factor"),
 IP("sens_area",DIO_AREA_SENS,IF_FLAG,"flag to request sensitivity WRT area"),
 OP("vd",DIO_VOLTAGE,IF_REAL, "Diode voltage"),
 OP("id",      DIO_CURRENT,IF_REAL, "Diode current"),
 OPR("c",     DIO_CURRENT,IF_REAL, "Diode current"),
 OP("gd",   DIO_CONDUCT,IF_REAL, "Diode conductance"),
 OP("cd", DIO_CAP, IF_REAL, "Diode capacitance"),
 OPU("charge", DIO_CHARGE, IF_REAL, "Diode capacitor charge"),
 OPU("capcur", DIO_CAPCUR, IF_REAL, "Diode capacitor current"),
 OPU("p",      DIO_POWER,  IF_REAL, "Diode power"),
 OPU("sens_dc",DIO_QUEST_SENS_DC,     IF_REAL,   "dc sensitivity "),
 OPU("sens_real", DIO_QUEST_SENS_REAL,IF_REAL,   
        "dc sens. and real part of ac sensitivity"),
 OPU("sens_imag", DIO_QUEST_SENS_IMAG,IF_REAL, "imag part of ac sensitivity "),
 OPU("sens_mag",  DIO_QUEST_SENS_MAG, IF_REAL, "sensitivity of ac magnitude"),
 OPU("sens_ph",   DIO_QUEST_SENS_PH,  IF_REAL, "sensitivity of ac phase"),
 OPU("sens_cplx", DIO_QUEST_SENS_CPLX,IF_COMPLEX,"ac sensitivity")
};

IFparm DIOmPTable[] = { /* model parameters */
 IOP( "is",  DIO_MOD_IS,  IF_REAL, "Saturation current"),
 IOPU( "tnom",DIO_MOD_TNOM,IF_REAL, "Parameter measurement temperature"),
 IOP( "rs",  DIO_MOD_RS,  IF_REAL, "Ohmic resistance"),
 IOP( "n",   DIO_MOD_N,   IF_REAL, "Emission Coefficient"),
 IOPA( "tt",  DIO_MOD_TT,  IF_REAL, "Transit Time"),
 IOPA( "cjo", DIO_MOD_CJO, IF_REAL, "Junction capacitance"),
 IOPR( "cj0", DIO_MOD_CJO, IF_REAL, "Junction capacitance"),
 IOP( "vj",  DIO_MOD_VJ,  IF_REAL, "Junction potential"),
 IOP( "m",   DIO_MOD_M,   IF_REAL, "Grading coefficient"),
 IOP( "eg",  DIO_MOD_EG,  IF_REAL, "Activation energy"),
 IOP( "xti", DIO_MOD_XTI, IF_REAL, "Saturation current temperature exp."),
 IOP( "kf",   DIO_MOD_KF,  IF_REAL, "flicker noise coefficient"),
 IOP( "af",   DIO_MOD_AF,  IF_REAL, "flicker noise exponent"),
 IOP( "fc",  DIO_MOD_FC,  IF_REAL, "Forward bias junction fit parameter"),
 IOP( "bv",  DIO_MOD_BV,  IF_REAL, "Reverse breakdown voltage"),
 IOP( "ibv", DIO_MOD_IBV, IF_REAL, "Current at reverse breakdown voltage"),
 OPU( "cond", DIO_MOD_COND,IF_REAL, "Ohmic conductance"),
 IP( "d",    DIO_MOD_D,   IF_FLAG, "Diode model")
};

char *DIOnames[] = {
    "D+",
    "D-"
};

int	DIOnSize = NUMELEMS(DIOnames);
int	DIOpTSize = NUMELEMS(DIOpTable);
int	DIOmPTSize = NUMELEMS(DIOmPTable);
int	DIOiSize = sizeof(DIOinstance);
int	DIOmSize = sizeof(DIOmodel);
