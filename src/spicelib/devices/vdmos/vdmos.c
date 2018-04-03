/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS Model: 2018 Holger Vogt
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "vdmosdefs.h"
#include "ngspice/suffix.h"

IFparm VDMOSpTable[] = { /* parameters */ 
 IOPU("m",            VDMOS_M,          IF_REAL,    "Multiplier"),
 IOPU("l",            VDMOS_L,          IF_REAL,    "Length"),
 IOPU("w",            VDMOS_W,          IF_REAL,    "Width"),
 IOPU("ad",           VDMOS_AD,         IF_REAL,    "Drain area"),
 IOPU("as",           VDMOS_AS,         IF_REAL,    "Source area"),
 IOPU("pd",           VDMOS_PD,         IF_REAL,    "Drain perimeter"),
 IOPU("ps",           VDMOS_PS,         IF_REAL,    "Source perimeter"),
 IOPU("nrd",          VDMOS_NRD,        IF_REAL,    "Drain squares"),
 IOPU("nrs",          VDMOS_NRS,        IF_REAL,    "Source squares"),
 IP("off",            VDMOS_OFF,        IF_FLAG,    "Device initially off"),
 IOPU("icvds",        VDMOS_IC_VDS,     IF_REAL,    "Initial D-S voltage"),
 IOPU("icvgs",        VDMOS_IC_VGS,     IF_REAL,    "Initial G-S voltage"),
 IOPU("icvbs",        VDMOS_IC_VBS,     IF_REAL,    "Initial B-S voltage"),
 IOPU("temp",         VDMOS_TEMP,       IF_REAL,    "Instance temperature"),
 IOPU("dtemp",        VDMOS_DTEMP,      IF_REAL,    "Instance temperature difference"),
 IP( "ic",            VDMOS_IC,         IF_REALVEC, "Vector of D-S, G-S, B-S voltages"),

 OP( "id",           VDMOS_CD,         IF_REAL,    "Drain current"),
 OP( "is",           VDMOS_CS,         IF_REAL,    "Source current"),
 OP( "ig",           VDMOS_CG,         IF_REAL,    "Gate current "),
 OP( "ib",           VDMOS_CB,         IF_REAL,    "Bulk current "),
 OPU( "ibd",         VDMOS_CBD,        IF_REAL,    "B-D junction current"),
 OPU( "ibs",         VDMOS_CBS,        IF_REAL,    "B-S junction current"),
 OP( "vgs",          VDMOS_VGS,        IF_REAL,    "Gate-Source voltage"),
 OP( "vds",          VDMOS_VDS,        IF_REAL,    "Drain-Source voltage"),
 OP( "vbs",          VDMOS_VBS,        IF_REAL,    "Bulk-Source voltage"),
 OPU( "vbd",         VDMOS_VBD,        IF_REAL,    "Bulk-Drain voltage"),
 /*
 OP( "cgs",          VDMOS_CGS,        IF_REAL,    "Gate-Source capacitance"),
 OP( "cgd",          VDMOS_CGD,        IF_REAL,    "Gate-Drain capacitance"),
 */

 OPU( "dnode",      VDMOS_DNODE,      IF_INTEGER, "Number of the drain node "),
 OPU( "gnode",      VDMOS_GNODE,      IF_INTEGER, "Number of the gate node "),
 OPU( "snode",      VDMOS_SNODE,      IF_INTEGER, "Number of the source node "),
 OPU( "bnode",      VDMOS_BNODE,      IF_INTEGER, "Number of the node "),
 OPU( "dnodeprime", VDMOS_DNODEPRIME, IF_INTEGER, "Number of int. drain node"),
 OPU( "snodeprime", VDMOS_SNODEPRIME, IF_INTEGER, "Number of int. source node "),

 OP( "von",               VDMOS_VON,           IF_REAL, " "),
 OP( "vdsat",             VDMOS_VDSAT,         IF_REAL, "Saturation drain voltage"),
 OPU( "sourcevcrit",      VDMOS_SOURCEVCRIT,   IF_REAL, "Critical source voltage"),
 OPU( "drainvcrit",       VDMOS_DRAINVCRIT,    IF_REAL, "Critical drain voltage"),
 OP( "rs",                VDMOS_SOURCERESIST,  IF_REAL, "Source resistance"),
 OPU("sourceconductance", VDMOS_SOURCECONDUCT, IF_REAL, "Conductance of source"),
 OP( "rd",                VDMOS_DRAINRESIST,   IF_REAL, "Drain conductance"),
 OPU("drainconductance",  VDMOS_DRAINCONDUCT,  IF_REAL, "Conductance of drain"),

 OP( "gm",        VDMOS_GM,         IF_REAL,    "Transconductance"),
 OP( "gds",       VDMOS_GDS,        IF_REAL,    "Drain-Source conductance"),
 OP( "gmb",       VDMOS_GMBS,       IF_REAL,    "Bulk-Source transconductance"),
 OPR( "gmbs",     VDMOS_GMBS,       IF_REAL,    ""),
 OPU( "gbd",      VDMOS_GBD,        IF_REAL,    "Bulk-Drain conductance"),
 OPU( "gbs",      VDMOS_GBS,        IF_REAL,    "Bulk-Source conductance"),

 OP( "cbd",        VDMOS_CAPBD,      IF_REAL,    "Bulk-Drain capacitance"),
 OP( "cbs",        VDMOS_CAPBS,      IF_REAL,    "Bulk-Source capacitance"),
 OP( "cgs",        VDMOS_CAPGS,      IF_REAL,    "Gate-Source capacitance"),
 OP( "cgd",        VDMOS_CAPGD,      IF_REAL,    "Gate-Drain capacitance"),
 OP( "cgb",        VDMOS_CAPGB,      IF_REAL,    "Gate-Bulk capacitance"),

 OPU( "cqgs", VDMOS_CQGS, IF_REAL, "Capacitance due to gate-source charge storage"),
 OPU( "cqgd", VDMOS_CQGD, IF_REAL, "Capacitance due to gate-drain charge storage"),
 OPU( "cqgb", VDMOS_CQGB, IF_REAL, "Capacitance due to gate-bulk charge storage"),
 OPU( "cqbd", VDMOS_CQBD, IF_REAL, "Capacitance due to bulk-drain charge storage"),
 OPU( "cqbs", VDMOS_CQBS, IF_REAL, "Capacitance due to bulk-source charge storage"),

 OPU( "qgs",      VDMOS_QGS,        IF_REAL,    "Gate-Source charge storage"),
 OPU( "qgd",      VDMOS_QGD,        IF_REAL,    "Gate-Drain charge storage"),
 OPU( "qgb",      VDMOS_QGB,        IF_REAL,    "Gate-Bulk charge storage"),
 OPU( "qbd",      VDMOS_QBD,        IF_REAL,    "Bulk-Drain charge storage"),
 OPU( "qbs",      VDMOS_QBS,        IF_REAL,    "Bulk-Source charge storage"),
 OPU( "p",        VDMOS_POWER,      IF_REAL,    "Instaneous power"),
};

IFparm VDMOSmPTable[] = { /* model parameters */
 OP("type",   VDMOS_MOD_TYPE,  IF_STRING, "N-channel or P-channel MOS"),
 IOP("vto",   VDMOS_MOD_VTO,   IF_REAL,   "Threshold voltage"),
 IOP("kp",    VDMOS_MOD_KP,    IF_REAL,   "Transconductance parameter"),
 IOP("phi",   VDMOS_MOD_PHI,   IF_REAL,   "Surface potential"),
 IOP("lambda",VDMOS_MOD_LAMBDA,IF_REAL,   "Channel length modulation"),
 IOP("rd",    VDMOS_MOD_RD,    IF_REAL,   "Drain ohmic resistance"),
 IOP("rs",    VDMOS_MOD_RS,    IF_REAL,   "Source ohmic resistance"),
 IOP("rg",    VDMOS_MOD_RG,    IF_REAL,   "Gate ohmic resistance"),
/*
 Cjo Zero-bias body diode junction capacitance
 Is Body diode saturation current
 N Bulk diode emission coefficient
 Vj Body diode junction potential
 M Body diode grading coefficient
 Fc Body diode coefficient for forward-bias depletion capacitance formula
 tt Body diode transit time
 Eg Body diode activation energy for temperature effect on Is
 Xti Body diode saturation current temperature exponent
*/

 /* gate-source and gate-drain capacitances */
 IOPA("cgdmin", VDMOS_MOD_CGDMIN, IF_REAL, "Minimum non-linear G-D capacitance"),
 IOPA("cgdmax", VDMOS_MOD_CGDMAX, IF_REAL, "Maximum non-linear G-D capacitance"),
 IOPA("a",      VDMOS_MOD_A,      IF_REAL, "Non-linear Cgd capacitance parameter"),
 IOPA("cgs",    VDMOS_MOD_CGS,    IF_REAL, "Gate-source capacitance"),

 IOP("tnom",  VDMOS_MOD_TNOM,  IF_REAL,   "Parameter measurement temperature"),
 IOP("kf",    VDMOS_MOD_KF,    IF_REAL,   "Flicker noise coefficient"),
 IOP("af",    VDMOS_MOD_AF,    IF_REAL,   "Flicker noise exponent"),
 IP("vdmosn", VDMOS_MOD_NMOS,  IF_FLAG,   "N type DMOSfet model"),
 IP("vdmosp", VDMOS_MOD_PMOS,  IF_FLAG,   "P type DMOSfet model"),
 IP("vdmos",  VDMOS_MOD_DMOS,  IF_REAL,   "DMOS transistor"),

/* MOS1 */
 IOPR("vt0",  VDMOS_MOD_VTO,   IF_REAL,   "Threshold voltage"),
 IOP("gamma", VDMOS_MOD_GAMMA, IF_REAL,   "Bulk threshold parameter"),
 IOPA("cbd",  VDMOS_MOD_CBD,   IF_REAL,   "B-D junction capacitance"),
 IOPA("cbs",  VDMOS_MOD_CBS,   IF_REAL,   "B-S junction capacitance"),
 IOP("is",    VDMOS_MOD_IS,    IF_REAL,   "Bulk junction sat. current"),
 IOP("pb",    VDMOS_MOD_PB,    IF_REAL,   "Bulk junction potential"),
 IOP("rsh",   VDMOS_MOD_RSH,   IF_REAL,   "Sheet resistance"),
 IOPA("cj",   VDMOS_MOD_CJ,    IF_REAL,   "Bottom junction cap per area"),
 IOP("mj",    VDMOS_MOD_MJ,    IF_REAL,   "Bottom grading coefficient"),
 IOPA("cjsw", VDMOS_MOD_CJSW,  IF_REAL,   "Side junction cap per area"),
 IOP("mjsw",  VDMOS_MOD_MJSW,  IF_REAL,   "Side grading coefficient"),
 IOP("js",    VDMOS_MOD_JS,    IF_REAL,   "Bulk jct. sat. current density"),
 IOP("tox",   VDMOS_MOD_TOX,   IF_REAL,   "Oxide thickness"),
 IOP("ld",    VDMOS_MOD_LD,    IF_REAL,   "Lateral diffusion"),
 IOP("u0",    VDMOS_MOD_U0,    IF_REAL,   "Surface mobility"),
 IOPR("uo",   VDMOS_MOD_U0,    IF_REAL,   "Surface mobility"),
 IOP("fc",    VDMOS_MOD_FC,    IF_REAL,   "Forward bias jct. fit parm."),
 IOP("nsub",  VDMOS_MOD_NSUB,  IF_REAL,   "Substrate doping"),
 IOP("tpg",   VDMOS_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nss",   VDMOS_MOD_NSS,   IF_REAL,   "Surface state density")
};

char *VDMOSnames[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int	VDMOSnSize = NUMELEMS(VDMOSnames);
int	VDMOSpTSize = NUMELEMS(VDMOSpTable);
int	VDMOSmPTSize = NUMELEMS(VDMOSmPTable);
int	VDMOSiSize = sizeof(VDMOSinstance);
int	VDMOSmSize = sizeof(VDMOSmodel);
