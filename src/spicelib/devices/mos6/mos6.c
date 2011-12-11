/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mos6defs.h"
#include "ngspice/suffix.h"

IFparm MOS6pTable[] = { /* parameters */ 
 IOPU("l",            MOS6_L,          IF_REAL   , "Length"),
 IOPU("w",            MOS6_W,          IF_REAL   , "Width"),
 IOPU("m",            MOS6_M,          IF_REAL   , "Parallel Multiplier"),
 IOPU("ad",           MOS6_AD,         IF_REAL   , "Drain area"),
 IOPU("as",           MOS6_AS,         IF_REAL   , "Source area"),
 IOPU("pd",           MOS6_PD,         IF_REAL   , "Drain perimeter"),
 IOPU("ps",           MOS6_PS,         IF_REAL   , "Source perimeter"),
 OP( "id",           MOS6_CD,         IF_REAL,    "Drain current"),
 OPR( "cd",           MOS6_CD,         IF_REAL,    "Drain current"),
 OP( "is",           MOS6_CS,         IF_REAL,    "Source current"),
 OP( "ig",           MOS6_CG,         IF_REAL,    "Gate current "),
 OP( "ib",           MOS6_CB,         IF_REAL,    "Bulk current "),
 OP( "ibs",      MOS6_CBS,    IF_REAL,    "B-S junction capacitance"),
 OP( "ibd",      MOS6_CBD,    IF_REAL,    "B-D junction capacitance"),
 OP( "vgs",          MOS6_VGS,        IF_REAL,    "Gate-Source voltage"),
 OP( "vds",          MOS6_VDS,        IF_REAL,    "Drain-Source voltage"),
 OP( "vbs",          MOS6_VBS,        IF_REAL,    "Bulk-Source voltage"),
 OPU( "vbd",          MOS6_VBD,        IF_REAL,    "Bulk-Drain voltage"),
 IOPU("nrd",          MOS6_NRD,        IF_REAL   , "Drain squares"),
 IOPU("nrs",          MOS6_NRS,        IF_REAL   , "Source squares"),
 IP("off",           MOS6_OFF,        IF_FLAG   , "Device initially off"),
 IOPAU("icvds",        MOS6_IC_VDS,     IF_REAL   , "Initial D-S voltage"),
 IOPAU("icvgs",        MOS6_IC_VGS,     IF_REAL   , "Initial G-S voltage"),
 IOPAU("icvbs",        MOS6_IC_VBS,     IF_REAL   , "Initial B-S voltage"),
 IOPU("temp",         MOS6_TEMP,       IF_REAL,    "Instance temperature"),
 IOPU("dtemp",        MOS6_DTEMP,      IF_REAL,    "Instance temperature difference"),
 IP( "ic",           MOS6_IC,  IF_REALVEC, "Vector of D-S, G-S, B-S voltages"),
 IP( "sens_l", MOS6_L_SENS, IF_FLAG, "flag to request sensitivity WRT length"),
 IP( "sens_w", MOS6_W_SENS, IF_FLAG, "flag to request sensitivity WRT width"),

/*
 OP( "cgs",          MOS6_CGS,        IF_REAL   , "Gate-Source capacitance"),
 OP( "cgd",          MOS6_CGD,        IF_REAL   , "Gate-Drain capacitance"),
*/

 OPU( "dnode",      MOS6_DNODE,      IF_INTEGER, "Number of the drain node "),
 OPU( "gnode",      MOS6_GNODE,      IF_INTEGER, "Number of the gate node "),
 OPU( "snode",      MOS6_SNODE,      IF_INTEGER, "Number of the source node "),
 OPU( "bnode",      MOS6_BNODE,      IF_INTEGER, "Number of the node "),
 OPU( "dnodeprime", MOS6_DNODEPRIME, IF_INTEGER, "Number of int. drain node"),
 OPU( "snodeprime", MOS6_SNODEPRIME, IF_INTEGER, "Number of int. source node "),
 OP( "rs", MOS6_SOURCERESIST, IF_REAL, "Source resistance"),
 OPU("sourceconductance", MOS6_SOURCECONDUCT, IF_REAL, "Source conductance"),
 OP( "rd",  MOS6_DRAINRESIST,  IF_REAL, "Drain resistance"),
 OPU( "drainconductance",  MOS6_DRAINCONDUCT,  IF_REAL, "Drain conductance"),
 OP( "von",          MOS6_VON,        IF_REAL,    "Turn-on voltage"),
 OP( "vdsat",        MOS6_VDSAT,      IF_REAL,    "Saturation drain voltage"),
 OPU( "sourcevcrit",  MOS6_SOURCEVCRIT,IF_REAL,    "Critical source voltage"),
 OPU( "drainvcrit",   MOS6_DRAINVCRIT, IF_REAL,    "Critical drain voltage"),

 OP( "gmbs",     MOS6_GMBS,   IF_REAL,    "Bulk-Source transconductance"),
 OP( "gm",           MOS6_GM,         IF_REAL,    "Transconductance"),
 OP( "gds",          MOS6_GDS,        IF_REAL,    "Drain-Source conductance"),
 OP( "gbd",          MOS6_GBD,        IF_REAL,    "Bulk-Drain conductance"),
 OP( "gbs",          MOS6_GBS,        IF_REAL,    "Bulk-Source conductance"),

 OP( "cgs",        MOS6_CAPGS,      IF_REAL,    "Gate-Source capacitance"),
 OP( "cgd",        MOS6_CAPGD,      IF_REAL,    "Gate-Drain capacitance"),
 OP( "cgb",        MOS6_CAPGB,      IF_REAL,    "Gate-Bulk capacitance"),
 OP( "cbd",        MOS6_CAPBD,      IF_REAL,    "Bulk-Drain capacitance"),
 OP( "cbs",        MOS6_CAPBS,      IF_REAL,    "Bulk-Source capacitance"),

 OP( "cbd0", MOS6_CAPZEROBIASBD, IF_REAL, "Zero-Bias B-D junction capacitance"),
 OP( "cbdsw0",        MOS6_CAPZEROBIASBDSW, IF_REAL,    " "),
 OP( "cbs0", MOS6_CAPZEROBIASBS, IF_REAL, "Zero-Bias B-S junction capacitance"),
 OP( "cbssw0",        MOS6_CAPZEROBIASBSSW, IF_REAL,    " "),

 OPU( "cqgs",MOS6_CQGS,IF_REAL,"Capacitance due to gate-source charge storage"),
 OPU( "cqgd",MOS6_CQGD,IF_REAL,"Capacitance due to gate-drain charge storage"),
 OPU( "cqgb",MOS6_CQGB,IF_REAL,"Capacitance due to gate-bulk charge storage"),
 OPU( "cqbd",MOS6_CQBD,IF_REAL,"Capacitance due to bulk-drain charge storage"),
 OPU( "cqbs",MOS6_CQBS,IF_REAL,"Capacitance due to bulk-source charge storage"),
 OPU( "qgs",        MOS6_QGS,        IF_REAL,    "Gate-Source charge storage"),
 OPU( "qgd",        MOS6_QGD,        IF_REAL,    "Gate-Drain charge storage"),
 OPU( "qgb",        MOS6_QGB,        IF_REAL,    "Gate-Bulk charge storage"),
 OPU( "qbd",        MOS6_QBD,        IF_REAL,    "Bulk-Drain charge storage"),
 OPU( "qbs",        MOS6_QBS,        IF_REAL,    "Bulk-Source charge storage"),
 OPU( "p",          MOS6_POWER,      IF_REAL,    "Instaneous power"),
 OPU( "sens_l_dc",    MOS6_L_SENS_DC,  IF_REAL,    "dc sensitivity wrt length"),
 OPU( "sens_l_real", MOS6_L_SENS_REAL,IF_REAL,
        "real part of ac sensitivity wrt length"),
 OPU( "sens_l_imag",  MOS6_L_SENS_IMAG,IF_REAL,    
        "imag part of ac sensitivity wrt length"),
 OPU( "sens_l_mag",   MOS6_L_SENS_MAG, IF_REAL,    
        "sensitivity wrt l of ac magnitude"),
 OPU( "sens_l_ph",    MOS6_L_SENS_PH,  IF_REAL,    
        "sensitivity wrt l of ac phase"),
 OPU( "sens_l_cplx",  MOS6_L_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt length"),
 OPU( "sens_w_dc",    MOS6_W_SENS_DC,  IF_REAL,    "dc sensitivity wrt width"),
 OPU( "sens_w_real",  MOS6_W_SENS_REAL,IF_REAL,    
        "real part of ac sensitivity wrt width"),
 OPU( "sens_w_imag",  MOS6_W_SENS_IMAG,IF_REAL,    
        "imag part of ac sensitivity wrt width"),
 OPU( "sens_w_mag",   MOS6_W_SENS_MAG, IF_REAL,    
        "sensitivity wrt w of ac magnitude"),
 OPU( "sens_w_ph",    MOS6_W_SENS_PH,  IF_REAL,    
        "sensitivity wrt w of ac phase"),
 OPU( "sens_w_cplx",  MOS6_W_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt width")
};

IFparm MOS6mPTable[] = { /* model parameters */
 OP("type",   MOS6_MOD_TYPE,   IF_STRING   ,"N-channel or P-channel MOS"),
 IOP("vto",   MOS6_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOPR("vt0",   MOS6_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOP("kv",    MOS6_MOD_KV,    IF_REAL   ,"Saturation voltage factor"),
 IOP("nv",    MOS6_MOD_NV,    IF_REAL   ,"Saturation voltage coeff."),
 IOP("kc",    MOS6_MOD_KC,    IF_REAL   ,"Saturation current factor"),
 IOP("nc",    MOS6_MOD_NC,    IF_REAL   ,"Saturation current coeff."),
 IOP("nvth",  MOS6_MOD_NVTH,  IF_REAL   ,"Threshold voltage coeff."),
 IOP("ps",    MOS6_MOD_PS,    IF_REAL   ,"Sat. current modification  par."),
 IOP("gamma", MOS6_MOD_GAMMA, IF_REAL   ,"Bulk threshold parameter"),
 IOP("gamma1",MOS6_MOD_GAMMA1,IF_REAL   ,"Bulk threshold parameter 1"),
 IOP("sigma", MOS6_MOD_SIGMA, IF_REAL   ,"Static feedback effect par."),
 IOP("phi",   MOS6_MOD_PHI,   IF_REAL   ,"Surface potential"),
 IOP("lambda",MOS6_MOD_LAMBDA,IF_REAL   ,"Channel length modulation param."),
 IOP("lambda0",MOS6_MOD_LAMDA0,IF_REAL   ,"Channel length modulation param. 0"),
 IOP("lambda1",MOS6_MOD_LAMDA1,IF_REAL   ,"Channel length modulation param. 1"),
 IOP("rd",    MOS6_MOD_RD,    IF_REAL   ,"Drain ohmic resistance"),
 IOP("rs",    MOS6_MOD_RS,    IF_REAL   ,"Source ohmic resistance"),
 IOPA("cbd",   MOS6_MOD_CBD,   IF_REAL   ,"B-D junction capacitance"),
 IOPA("cbs",   MOS6_MOD_CBS,   IF_REAL   ,"B-S junction capacitance"),
 IOP("is",    MOS6_MOD_IS,    IF_REAL   ,"Bulk junction sat. current"),
 IOP("pb",    MOS6_MOD_PB,    IF_REAL   ,"Bulk junction potential"),
 IOPA("cgso",  MOS6_MOD_CGSO,  IF_REAL   ,"Gate-source overlap cap."),
 IOPA("cgdo",  MOS6_MOD_CGDO,  IF_REAL   ,"Gate-drain overlap cap."),
 IOPA("cgbo",  MOS6_MOD_CGBO,  IF_REAL   ,"Gate-bulk overlap cap."),
 IOP("rsh",   MOS6_MOD_RSH,   IF_REAL   ,"Sheet resistance"),
 IOPA("cj",    MOS6_MOD_CJ,    IF_REAL   ,"Bottom junction cap per area"),
 IOP("mj",    MOS6_MOD_MJ,    IF_REAL   ,"Bottom grading coefficient"),
 IOPA("cjsw",  MOS6_MOD_CJSW,  IF_REAL   ,"Side junction cap per area"),
 IOP("mjsw",  MOS6_MOD_MJSW,  IF_REAL   ,"Side grading coefficient"),
 IOP("js",    MOS6_MOD_JS,    IF_REAL   ,"Bulk jct. sat. current density"),
 IOP("ld",    MOS6_MOD_LD,    IF_REAL   ,"Lateral diffusion"),
 IOP("tox",   MOS6_MOD_TOX,   IF_REAL   ,"Oxide thickness"),
 IOP("u0",    MOS6_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOPR("uo",    MOS6_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOP("fc",    MOS6_MOD_FC,    IF_REAL   ,"Forward bias jct. fit parm."),
 IP("nmos",   MOS6_MOD_NMOS,  IF_FLAG   ,"N type MOSfet model"),
 IP("pmos",   MOS6_MOD_PMOS,  IF_FLAG   ,"P type MOSfet model"),
 IOP("tpg",   MOS6_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nsub",  MOS6_MOD_NSUB,  IF_REAL   ,"Substrate doping"),
 IOP("nss",   MOS6_MOD_NSS,   IF_REAL   ,"Surface state density"),
 IOP("tnom",  MOS6_MOD_TNOM,  IF_REAL   ,"Parameter measurement temperature")
};

char *MOS6names[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int     MOS6nSize = NUMELEMS(MOS6names);
int     MOS6pTSize = NUMELEMS(MOS6pTable);
int     MOS6mPTSize = NUMELEMS(MOS6mPTable);
int	MOS6iSize = sizeof(MOS6instance);
int	MOS6mSize = sizeof(MOS6model);
