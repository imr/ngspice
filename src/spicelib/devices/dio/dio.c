/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "diodefs.h"
#include "ngspice/suffix.h"

IFparm DIOpTable[] = { /* parameters */ 
 IOPU("off",   DIO_OFF,   IF_FLAG, "Initially off"),
 IOPU("temp",  DIO_TEMP,  IF_REAL, "Instance temperature"),
 IOPU("dtemp", DIO_DTEMP, IF_REAL, "Instance delta temperature"),
 IOPAU("ic",   DIO_IC,    IF_REAL, "Initial device voltage"),
 IOPU("area",  DIO_AREA,  IF_REAL, "Area factor"),
 IOPU("pj",    DIO_PJ,    IF_REAL, "Perimeter factor"),
 IOPU("w",     DIO_W,     IF_REAL, "Diode width"),
 IOPU("l",     DIO_L,     IF_REAL, "Diode length"),
 IOPU("m",     DIO_M,     IF_REAL, "Multiplier"),
 IOPU("lm",    DIO_LM,    IF_REAL, "Length of metal capacitor (level=3)"),
 IOPU("lp",    DIO_LP,    IF_REAL, "Length of polysilicon capacitor (level=3)"),
 IOPU("wm",    DIO_WM,    IF_REAL, "Width of metal capacitor (level=3)"),
 IOPU("wp",    DIO_WP,    IF_REAL, "Width of polysilicon capacitor (level=3)"),
 IOP("thermal",DIO_THERMAL, IF_FLAG, "Self heating mode selector"),

 IP("sens_area",DIO_AREA_SENS,IF_FLAG,"flag to request sensitivity WRT area"),
 OP("vd",      DIO_VOLTAGE,IF_REAL, "Diode voltage"),
 OP("id",      DIO_CURRENT,IF_REAL, "Diode current"),
 OPR("c",     DIO_CURRENT,IF_REAL, "Diode current"),
 OP("gd",   DIO_CONDUCT,IF_REAL, "Diode conductance"),
 OP("cd",   DIO_CAP, IF_REAL, "Diode capacitance"),
 OPU("charge", DIO_CHARGE, IF_REAL, "Diode capacitor charge"),
 OPUR("qd", DIO_CHARGE, IF_REAL, "Diode capacitor charge"),
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
 IOP( "level", DIO_MOD_LEVEL, IF_INTEGER, "Diode level selector"),
 IOP(  "is",  DIO_MOD_IS,  IF_REAL, "Saturation current"),
 IOPR( "js",  DIO_MOD_IS,  IF_REAL, "Saturation current"),
 IOP( "jsw", DIO_MOD_JSW,  IF_REAL, "Sidewall Saturation current"),

 IOPU( "tnom",DIO_MOD_TNOM,IF_REAL, "Parameter measurement temperature"),
 IOPUR("tref",DIO_MOD_TNOM,IF_REAL, "Parameter measurement temperature"),
 IOP( "rs",  DIO_MOD_RS,  IF_REAL, "Ohmic resistance"),
 IOP( "trs", DIO_MOD_TRS, IF_REAL, "Ohmic resistance 1st order temp. coeff."),
 IOPR( "trs1", DIO_MOD_TRS, IF_REAL, "Ohmic resistance 1st order temp. coeff."),
 IOP( "trs2", DIO_MOD_TRS2, IF_REAL, "Ohmic resistance 2nd order temp. coeff."),
 IOP( "n",   DIO_MOD_N,   IF_REAL, "Emission Coefficient"),
 IOP( "ns",   DIO_MOD_NS,   IF_REAL, "Sidewall emission Coefficient"),
 IOPA( "tt",  DIO_MOD_TT,  IF_REAL, "Transit Time"),
 IOPA( "ttt1", DIO_MOD_TTT1, IF_REAL, "Transit Time 1st order temp. coeff."),
 IOPA( "ttt2", DIO_MOD_TTT2, IF_REAL, "Transit Time 2nd order temp. coeff."),
 IOPA( "cjo", DIO_MOD_CJO, IF_REAL, "Junction capacitance"),
 IOPAR( "cj0",DIO_MOD_CJO, IF_REAL, "Junction capacitance"),
 IOPAR( "cj", DIO_MOD_CJO, IF_REAL, "Junction capacitance"),
 IOP( "vj",  DIO_MOD_VJ,  IF_REAL, "Junction potential"),
 IOPR( "pb",  DIO_MOD_VJ,  IF_REAL, "Junction potential"),
 IOP( "m",   DIO_MOD_M,   IF_REAL, "Grading coefficient"),
 IOPR( "mj",  DIO_MOD_M,   IF_REAL, "Grading coefficient"),
 IOP( "tm1", DIO_MOD_TM1,  IF_REAL, "Grading coefficient 1st temp. coeff."),
 IOP( "tm2", DIO_MOD_TM2,  IF_REAL, "Grading coefficient 2nd temp. coeff."),
 IOP( "cjp", DIO_MOD_CJSW, IF_REAL, "Sidewall junction capacitance"),
 IOPR( "cjsw", DIO_MOD_CJSW, IF_REAL, "Sidewall junction capacitance"),
 IOP( "php",  DIO_MOD_VJSW,  IF_REAL, "Sidewall junction potential"),
 IOP( "mjsw",  DIO_MOD_MJSW,   IF_REAL, "Sidewall Grading coefficient"),
 IOP( "ikf",  DIO_MOD_IKF,   IF_REAL, "Forward Knee current"),
 IOPR( "ik",  DIO_MOD_IKF,   IF_REAL, "Forward Knee current"),
 IOP( "ikr",  DIO_MOD_IKR,   IF_REAL, "Reverse Knee current"),
 IOP( "nbv",  DIO_MOD_NBV,   IF_REAL, "Breakdown Emission Coefficient"),
 IOP("area",  DIO_MOD_AREA,  IF_REAL, "Area factor"),
 IOP( "pj",   DIO_MOD_PJ,    IF_REAL, "Perimeter factor"),

 IOP( "tlev", DIO_MOD_TLEV, IF_INTEGER, "Diode temperature equation selector"),
 IOP( "tlevc", DIO_MOD_TLEVC, IF_INTEGER, "Diode temperature equation selector"),
 IOP( "eg",  DIO_MOD_EG,  IF_REAL, "Activation energy"),
 IOP( "xti", DIO_MOD_XTI, IF_REAL, "Saturation current temperature exp."),
 IOP( "cta", DIO_MOD_CTA, IF_REAL, "Area junction temperature coefficient"),
 IOPR( "ctc", DIO_MOD_CTA, IF_REAL, "Area junction capacitance temperature coefficient"),
 IOP( "ctp", DIO_MOD_CTP, IF_REAL, "Perimeter junction capacitance temperature coefficient"),

 IOP( "tpb", DIO_MOD_TPB, IF_REAL, "Area junction potential temperature coefficient"),
 IOPR( "tvj", DIO_MOD_TPB, IF_REAL, "Area junction potential temperature coefficient"),
 IOP( "tphp", DIO_MOD_TPHP, IF_REAL, "Perimeter junction potential temperature coefficient"),

 IOP( "jtun",  DIO_MOD_JTUN, IF_REAL, "Tunneling saturation current"),
 IOP( "jtunsw", DIO_MOD_JTUNSW, IF_REAL, "Tunneling sidewall saturation current"),
 IOP( "ntun",  DIO_MOD_NTUN, IF_REAL, "Tunneling emission coefficient"),
 IOP( "xtitun", DIO_MOD_XTITUN, IF_REAL, "Tunneling saturation current exponential"),
 IOP( "keg", DIO_MOD_KEG, IF_REAL, "EG correction factor for tunneling"),

 IOP( "kf",   DIO_MOD_KF,  IF_REAL, "flicker noise coefficient"),
 IOP( "af",   DIO_MOD_AF,  IF_REAL, "flicker noise exponent"),
 IOP( "fc",  DIO_MOD_FC,  IF_REAL, "Forward bias junction fit parameter"),
 IOP( "fcs",  DIO_MOD_FCS,  IF_REAL, "Forward bias sidewall junction fit parameter"),
 IOP( "bv",  DIO_MOD_BV,  IF_REAL, "Reverse breakdown voltage"),
 IOP( "ibv", DIO_MOD_IBV, IF_REAL, "Current at reverse breakdown voltage"),
 IOPR( "ib", DIO_MOD_IBV, IF_REAL, "Current at reverse breakdown voltage"),
 IOP( "tcv", DIO_MOD_TCV, IF_REAL, "Reverse breakdown voltage temperature coefficient"),
 OPU( "cond", DIO_MOD_COND,IF_REAL, "Ohmic conductance"),
 IOP( "isr",  DIO_MOD_ISR,  IF_REAL, "Recombination saturation current"),
 IOP( "nr",   DIO_MOD_NR,   IF_REAL, "Recombination current emission coefficient"),

 /* SOA parameters */
 IOP( "fv_max",   DIO_MOD_FV_MAX,  IF_REAL, "maximum voltage in forward direction"),
 IOP( "bv_max",   DIO_MOD_BV_MAX,  IF_REAL, "maximum voltage in reverse direction"),
 IOP( "id_max",   DIO_MOD_ID_MAX,  IF_REAL, "maximum current"),
 IOP( "te_max",   DIO_MOD_TE_MAX,  IF_REAL, "temperature"),
 IOP( "pd_max",   DIO_MOD_PD_MAX,  IF_REAL, "maximum power dissipation"),

/* self heating */
 IOP("rth0",  DIO_MOD_RTH0,  IF_REAL, "Self-heating thermal resistance"),
 IOP("cth0",  DIO_MOD_CTH0,  IF_REAL, "Self-heating thermal capacitance"),
/* scaled parasitic capacitances level 3 model */
 IOP( "lm",  DIO_MOD_LM,  IF_REAL, "Length of metal capacitor (level=3)"),
 IOP( "lp",  DIO_MOD_LP,  IF_REAL, "Length of polysilicon capacitor (level=3)"),
 IOP( "wm",  DIO_MOD_WM,  IF_REAL, "Width of metal capacitor (level=3)"),
 IOP( "wp",  DIO_MOD_WP,  IF_REAL, "Width of polysilicon capacitor (level=3)"),
 IOP( "xom", DIO_MOD_XOM, IF_REAL, "Thickness of the metal to bulk oxide (level=3)"),
 IOP( "xoi", DIO_MOD_XOI, IF_REAL, "Thickness of the polysilicon to bulk oxide (level=3)"),
 IOP( "xm",  DIO_MOD_XM,  IF_REAL, "Masking and etching effects in metal (level=3)"),
 IOP( "xp",  DIO_MOD_XP,  IF_REAL, "Masking and etching effects in polysilicon (level=3)"),

 IP( "d",    DIO_MOD_D,   IF_FLAG, "Diode model")
};

char *DIOnames[] = {
    "D+",
    "D-",
    "Tj"
};

int DIOnSize = NUMELEMS(DIOnames);
int DIOpTSize = NUMELEMS(DIOpTable);
int DIOmPTSize = NUMELEMS(DIOmPTable);
int DIOiSize = sizeof(DIOinstance);
int DIOmSize = sizeof(DIOmodel);
