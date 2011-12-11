/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mos1defs.h"
#include "ngspice/suffix.h"

IFparm MOS1pTable[] = { /* parameters */ 
 IOPU("m",            MOS1_M,          IF_REAL   , "Multiplier"),
 IOPU("l",            MOS1_L,          IF_REAL   , "Length"),
 IOPU("w",            MOS1_W,          IF_REAL   , "Width"),
 IOPU("ad",           MOS1_AD,         IF_REAL   , "Drain area"),
 IOPU("as",           MOS1_AS,         IF_REAL   , "Source area"),
 IOPU("pd",           MOS1_PD,         IF_REAL   , "Drain perimeter"),
 IOPU("ps",           MOS1_PS,         IF_REAL   , "Source perimeter"),
 IOPU("nrd",          MOS1_NRD,        IF_REAL   , "Drain squares"),
 IOPU("nrs",          MOS1_NRS,        IF_REAL   , "Source squares"),
 IP("off",           MOS1_OFF,        IF_FLAG   , "Device initially off"),
 IOPU("icvds",        MOS1_IC_VDS,     IF_REAL   , "Initial D-S voltage"),
 IOPU("icvgs",        MOS1_IC_VGS,     IF_REAL   , "Initial G-S voltage"),
 IOPU("icvbs",        MOS1_IC_VBS,     IF_REAL   , "Initial B-S voltage"),
 IOPU("temp",         MOS1_TEMP,       IF_REAL,    "Instance temperature"),
 IOPU("dtemp",         MOS1_DTEMP,       IF_REAL,    "Instance temperature difference"),
 IP( "ic",           MOS1_IC,  IF_REALVEC, "Vector of D-S, G-S, B-S voltages"),
 IP( "sens_l", MOS1_L_SENS, IF_FLAG, "flag to request sensitivity WRT length"),
 IP( "sens_w", MOS1_W_SENS, IF_FLAG, "flag to request sensitivity WRT width"),

 OP( "id",           MOS1_CD,         IF_REAL,    "Drain current"),
 OP( "is",           MOS1_CS,         IF_REAL,    "Source current"),
 OP( "ig",           MOS1_CG,         IF_REAL,    "Gate current "),
 OP( "ib",           MOS1_CB,         IF_REAL,    "Bulk current "),
 OPU( "ibd",      MOS1_CBD,    IF_REAL,    "B-D junction current"),
 OPU( "ibs",      MOS1_CBS,    IF_REAL,    "B-S junction current"),
 OP( "vgs",          MOS1_VGS,        IF_REAL,    "Gate-Source voltage"),
 OP( "vds",          MOS1_VDS,        IF_REAL,    "Drain-Source voltage"),
 OP( "vbs",          MOS1_VBS,        IF_REAL,    "Bulk-Source voltage"),
 OPU( "vbd",          MOS1_VBD,        IF_REAL,    "Bulk-Drain voltage"),
 /*
 OP( "cgs",          MOS1_CGS,        IF_REAL   , "Gate-Source capacitance"),
 OP( "cgd",          MOS1_CGD,        IF_REAL   , "Gate-Drain capacitance"),
 */

 OPU( "dnode",      MOS1_DNODE,      IF_INTEGER, "Number of the drain node "),
 OPU( "gnode",      MOS1_GNODE,      IF_INTEGER, "Number of the gate node "),
 OPU( "snode",      MOS1_SNODE,      IF_INTEGER, "Number of the source node "),
 OPU( "bnode",      MOS1_BNODE,      IF_INTEGER, "Number of the node "),
 OPU( "dnodeprime", MOS1_DNODEPRIME, IF_INTEGER, "Number of int. drain node"),
 OPU( "snodeprime", MOS1_SNODEPRIME, IF_INTEGER, "Number of int. source node "),

 OP( "von",          MOS1_VON,        IF_REAL,    " "),
 OP( "vdsat",        MOS1_VDSAT,      IF_REAL,    "Saturation drain voltage"),
 OPU( "sourcevcrit",  MOS1_SOURCEVCRIT,IF_REAL,    "Critical source voltage"),
 OPU( "drainvcrit",   MOS1_DRAINVCRIT, IF_REAL,    "Critical drain voltage"),
 OP( "rs", MOS1_SOURCERESIST, IF_REAL, "Source resistance"),
 OPU("sourceconductance", MOS1_SOURCECONDUCT, IF_REAL, "Conductance of source"),
 OP( "rd",  MOS1_DRAINRESIST,  IF_REAL, "Drain conductance"),
 OPU("drainconductance",  MOS1_DRAINCONDUCT,  IF_REAL, "Conductance of drain"),

 OP( "gm",           MOS1_GM,         IF_REAL,    "Transconductance"),
 OP( "gds",          MOS1_GDS,        IF_REAL,    "Drain-Source conductance"),
 OP( "gmb",     MOS1_GMBS,   IF_REAL,    "Bulk-Source transconductance"),
 OPR( "gmbs",     MOS1_GMBS,   IF_REAL,    ""),
 OPU( "gbd",          MOS1_GBD,        IF_REAL,    "Bulk-Drain conductance"),
 OPU( "gbs",          MOS1_GBS,        IF_REAL,    "Bulk-Source conductance"),

 OP( "cbd",        MOS1_CAPBD,      IF_REAL,    "Bulk-Drain capacitance"),
 OP( "cbs",        MOS1_CAPBS,      IF_REAL,    "Bulk-Source capacitance"),
 OP( "cgs",        MOS1_CAPGS,      IF_REAL,    "Gate-Source capacitance"),
 OP( "cgd",        MOS1_CAPGD,      IF_REAL,    "Gate-Drain capacitance"),
 OP( "cgb",        MOS1_CAPGB,      IF_REAL,    "Gate-Bulk capacitance"),

 OPU( "cqgs",MOS1_CQGS,IF_REAL,"Capacitance due to gate-source charge storage"),
 OPU( "cqgd",MOS1_CQGD,IF_REAL,"Capacitance due to gate-drain charge storage"),
 OPU( "cqgb",MOS1_CQGB,IF_REAL,"Capacitance due to gate-bulk charge storage"),
 OPU( "cqbd",MOS1_CQBD,IF_REAL,"Capacitance due to bulk-drain charge storage"),
 OPU( "cqbs",MOS1_CQBS,IF_REAL,"Capacitance due to bulk-source charge storage"),

 OP( "cbd0", MOS1_CAPZEROBIASBD, IF_REAL, "Zero-Bias B-D junction capacitance"),
 OP( "cbdsw0",        MOS1_CAPZEROBIASBDSW, IF_REAL,    " "),
 OP( "cbs0", MOS1_CAPZEROBIASBS, IF_REAL, "Zero-Bias B-S junction capacitance"),
 OP( "cbssw0",        MOS1_CAPZEROBIASBSSW, IF_REAL,    " "),

 OPU( "qgs",         MOS1_QGS,        IF_REAL,    "Gate-Source charge storage"),
 OPU( "qgd",         MOS1_QGD,        IF_REAL,    "Gate-Drain charge storage"),
 OPU( "qgb",         MOS1_QGB,        IF_REAL,    "Gate-Bulk charge storage"),
 OPU( "qbd",         MOS1_QBD,        IF_REAL,    "Bulk-Drain charge storage"),
 OPU( "qbs",         MOS1_QBS,        IF_REAL,    "Bulk-Source charge storage"),
 OPU( "p",            MOS1_POWER,      IF_REAL,    "Instaneous power"),
 OPU( "sens_l_dc",    MOS1_L_SENS_DC,  IF_REAL,    "dc sensitivity wrt length"),
 OPU( "sens_l_real", MOS1_L_SENS_REAL,IF_REAL,
        "real part of ac sensitivity wrt length"),
 OPU( "sens_l_imag",  MOS1_L_SENS_IMAG,IF_REAL,    
        "imag part of ac sensitivity wrt length"),
 OPU( "sens_l_mag",   MOS1_L_SENS_MAG, IF_REAL,    
        "sensitivity wrt l of ac magnitude"),
 OPU( "sens_l_ph",    MOS1_L_SENS_PH,  IF_REAL,    
        "sensitivity wrt l of ac phase"),
 OPU( "sens_l_cplx",  MOS1_L_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt length"),
 OPU( "sens_w_dc",    MOS1_W_SENS_DC,  IF_REAL,    "dc sensitivity wrt width"),
 OPU( "sens_w_real",  MOS1_W_SENS_REAL,IF_REAL,    
        "real part of ac sensitivity wrt width"),
 OPU( "sens_w_imag",  MOS1_W_SENS_IMAG,IF_REAL,    
        "imag part of ac sensitivity wrt width"),
 OPU( "sens_w_mag",   MOS1_W_SENS_MAG, IF_REAL,    
        "sensitivity wrt w of ac magnitude"),
 OPU( "sens_w_ph",    MOS1_W_SENS_PH,  IF_REAL,    
        "sensitivity wrt w of ac phase"),
 OPU( "sens_w_cplx",  MOS1_W_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt width")
};

IFparm MOS1mPTable[] = { /* model parameters */
 OP("type",   MOS1_MOD_TYPE,  IF_STRING, "N-channel or P-channel MOS"),
 IOP("vto",   MOS1_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOPR("vt0",  MOS1_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOP("kp",    MOS1_MOD_KP,    IF_REAL   ,"Transconductance parameter"),
 IOP("gamma", MOS1_MOD_GAMMA, IF_REAL   ,"Bulk threshold parameter"),
 IOP("phi",   MOS1_MOD_PHI,   IF_REAL   ,"Surface potential"),
 IOP("lambda",MOS1_MOD_LAMBDA,IF_REAL   ,"Channel length modulation"),
 IOP("rd",    MOS1_MOD_RD,    IF_REAL   ,"Drain ohmic resistance"),
 IOP("rs",    MOS1_MOD_RS,    IF_REAL   ,"Source ohmic resistance"),
 IOPA("cbd",  MOS1_MOD_CBD,   IF_REAL   ,"B-D junction capacitance"),
 IOPA("cbs",  MOS1_MOD_CBS,   IF_REAL   ,"B-S junction capacitance"),
 IOP("is",    MOS1_MOD_IS,    IF_REAL   ,"Bulk junction sat. current"),
 IOP("pb",    MOS1_MOD_PB,    IF_REAL   ,"Bulk junction potential"),
 IOPA("cgso", MOS1_MOD_CGSO,  IF_REAL   ,"Gate-source overlap cap."),
 IOPA("cgdo", MOS1_MOD_CGDO,  IF_REAL   ,"Gate-drain overlap cap."),
 IOPA("cgbo", MOS1_MOD_CGBO,  IF_REAL   ,"Gate-bulk overlap cap."),
 IOP("rsh",   MOS1_MOD_RSH,   IF_REAL   ,"Sheet resistance"),
 IOPA("cj",   MOS1_MOD_CJ,    IF_REAL   ,"Bottom junction cap per area"),
 IOP("mj",    MOS1_MOD_MJ,    IF_REAL   ,"Bottom grading coefficient"),
 IOPA("cjsw", MOS1_MOD_CJSW,  IF_REAL   ,"Side junction cap per area"),
 IOP("mjsw",  MOS1_MOD_MJSW,  IF_REAL   ,"Side grading coefficient"),
 IOP("js",    MOS1_MOD_JS,    IF_REAL   ,"Bulk jct. sat. current density"),
 IOP("tox",   MOS1_MOD_TOX,   IF_REAL   ,"Oxide thickness"),
 IOP("ld",    MOS1_MOD_LD,    IF_REAL   ,"Lateral diffusion"),
 IOP("u0",    MOS1_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOPR("uo",   MOS1_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOP("fc",    MOS1_MOD_FC,    IF_REAL   ,"Forward bias jct. fit parm."),
 IP("nmos",   MOS1_MOD_NMOS,  IF_FLAG   ,"N type MOSfet model"),
 IP("pmos",   MOS1_MOD_PMOS,  IF_FLAG   ,"P type MOSfet model"),
 IOP("nsub",  MOS1_MOD_NSUB,  IF_REAL   ,"Substrate doping"),
 IOP("tpg",   MOS1_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nss",   MOS1_MOD_NSS,   IF_REAL   ,"Surface state density"),
 IOP("tnom",  MOS1_MOD_TNOM,  IF_REAL   ,"Parameter measurement temperature"),
 IOP("kf",     MOS1_MOD_KF,    IF_REAL   ,"Flicker noise coefficient"),
 IOP("af",     MOS1_MOD_AF,    IF_REAL   ,"Flicker noise exponent")
};

char *MOS1names[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int	MOS1nSize = NUMELEMS(MOS1names);
int	MOS1pTSize = NUMELEMS(MOS1pTable);
int	MOS1mPTSize = NUMELEMS(MOS1mPTable);
int	MOS1iSize = sizeof(MOS1instance);
int	MOS1mSize = sizeof(MOS1model);
