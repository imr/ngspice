/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mos9defs.h"
#include "ngspice/suffix.h"

IFparm MOS9pTable[] = { /* parameters */ 

 IOPU("m",         MOS9_M,       IF_REAL   , "Multiplier"),
 IOPU("l",         MOS9_L,       IF_REAL   , "Length"),
 IOPU("w",         MOS9_W,       IF_REAL   , "Width"),
 IOPU("ad",        MOS9_AD,      IF_REAL   , "Drain area"),
 IOPU("as",        MOS9_AS,      IF_REAL   , "Source area"),
 IOPU("pd",        MOS9_PD,      IF_REAL   , "Drain perimeter"),
 IOPU("ps",        MOS9_PS,      IF_REAL   , "Source perimeter"),
 OP("id",    MOS9_CD,            IF_REAL, "Drain current"),
 OPR("cd",    MOS9_CD,            IF_REAL, "Drain current"),
 OPU("ibd",   MOS9_CBD,           IF_REAL, "B-D junction current"),
 OPU("ibs",   MOS9_CBS,           IF_REAL, "B-S junction current"),
 OPU("is",   MOS9_CS,    IF_REAL, "Source current"),
 OPU("ig",   MOS9_CG,    IF_REAL, "Gate current"),
 OPU("ib",   MOS9_CB,    IF_REAL, "Bulk current"),
 OP("vgs",   MOS9_VGS,            IF_REAL, "Gate-Source voltage"),
 OP("vds",   MOS9_VDS,            IF_REAL, "Drain-Source voltage"),
 OP("vbs",   MOS9_VBS,            IF_REAL, "Bulk-Source voltage"),
 OPU("vbd",   MOS9_VBD,            IF_REAL, "Bulk-Drain voltage"),
 IOPU("nrd",       MOS9_NRD,     IF_REAL   , "Drain squares"),
 IOPU("nrs",       MOS9_NRS,     IF_REAL   , "Source squares"),
 IP("off",        MOS9_OFF,     IF_FLAG   , "Device initially off"),
 IOPAU("icvds",       MOS9_IC_VDS,  IF_REAL   , "Initial D-S voltage"),
 IOPAU("icvgs",       MOS9_IC_VGS,  IF_REAL   , "Initial G-S voltage"),
 IOPAU("icvbs",       MOS9_IC_VBS,  IF_REAL   , "Initial B-S voltage"),
 IOPU("ic",     MOS9_IC,      IF_REALVEC, "Vector of D-S, G-S, B-S voltages"),
 IOPU("temp",      MOS9_TEMP,    IF_REAL   , "Instance operating temperature"),
 IOPU("dtemp",      MOS9_DTEMP,    IF_REAL   , "Instance operating temperature difference"),
 IP("sens_l",  MOS9_L_SENS, IF_FLAG, "flag to request sensitivity WRT length"),
 IP("sens_w",  MOS9_W_SENS, IF_FLAG, "flag to request sensitivity WRT width"),
 OPU("dnode",     MOS9_DNODE,   IF_INTEGER, "Number of drain node"),
 OPU("gnode",     MOS9_GNODE,   IF_INTEGER, "Number of gate node"),
 OPU("snode",     MOS9_SNODE,   IF_INTEGER, "Number of source node"),
 OPU("bnode",     MOS9_BNODE,   IF_INTEGER, "Number of bulk node"),
 OPU("dnodeprime", MOS9_DNODEPRIME,IF_INTEGER,"Number of internal drain node"),
 OPU("snodeprime", MOS9_SNODEPRIME,IF_INTEGER,"Number of internal source node"),
 OP("von",               MOS9_VON,           IF_REAL,    "Turn-on voltage"),
 OP("vdsat",       MOS9_VDSAT,         IF_REAL, "Saturation drain voltage"),
 OPU("sourcevcrit", MOS9_SOURCEVCRIT,   IF_REAL, "Critical source voltage"),
 OPU("drainvcrit",  MOS9_DRAINVCRIT,    IF_REAL, "Critical drain voltage"),
 OP("rs", MOS9_SOURCERESIST, IF_REAL,  "Source resistance"),
 OPU("sourceconductance", MOS9_SOURCECONDUCT, IF_REAL,  "Source conductance"),
 OP("rd",  MOS9_DRAINRESIST,  IF_REAL,  "Drain resistance"),
 OPU("drainconductance",  MOS9_DRAINCONDUCT,  IF_REAL,  "Drain conductance"),
 OP("gm",    MOS9_GM,            IF_REAL, "Transconductance"),
 OP("gds",   MOS9_GDS,           IF_REAL, "Drain-Source conductance"),
 OP("gmb",  MOS9_GMBS,           IF_REAL, "Bulk-Source transconductance"),
 OPR("gmbs",  MOS9_GMBS,         IF_REAL, "Bulk-Source transconductance"),
 OPU("gbd",   MOS9_GBD,           IF_REAL, "Bulk-Drain conductance"),
 OPU("gbs",   MOS9_GBS,           IF_REAL, "Bulk-Source conductance"),

 OP("cbd", MOS9_CAPBD,         IF_REAL, "Bulk-Drain capacitance"),
 OP("cbs", MOS9_CAPBS,         IF_REAL, "Bulk-Source capacitance"),
 OP("cgs", MOS9_CAPGS,         IF_REAL, "Gate-Source capacitance"),
/* OPR("cgs",       MOS9_CGS,     IF_REAL   , "Gate-Source capacitance"),*/
 OP("cgd", MOS9_CAPGD,         IF_REAL, "Gate-Drain capacitance"),
/* OPR("cgd",       MOS9_CGD,     IF_REAL   , "Gate-Drain capacitance"),*/
 OP("cgb", MOS9_CAPGB,	       IF_REAL, "Gate-Bulk capacitance"),

 OPU("cqgs",MOS9_CQGS,IF_REAL,"Capacitance due to gate-source charge storage"),
 OPU("cqgd",MOS9_CQGD, IF_REAL,"Capacitance due to gate-drain charge storage"),
 OPU("cqgb",MOS9_CQGB,  IF_REAL,"Capacitance due to gate-bulk charge storage"),
 OPU("cqbd",MOS9_CQBD,IF_REAL,"Capacitance due to bulk-drain charge storage"),
 OPU("cqbs",MOS9_CQBS,IF_REAL,"Capacitance due to bulk-source charge storage"),

 OPU("cbd0",MOS9_CAPZEROBIASBD,IF_REAL,"Zero-Bias B-D junction capacitance"),
 OPU("cbdsw0",MOS9_CAPZEROBIASBDSW,IF_REAL,
					"Zero-Bias B-D sidewall capacitance"),
 OPU("cbs0",MOS9_CAPZEROBIASBS,IF_REAL,"Zero-Bias B-S junction capacitance"),
 OPU("cbssw0",MOS9_CAPZEROBIASBSSW,IF_REAL,
					"Zero-Bias B-S sidewall capacitance"),
 OPU("qbs",  MOS9_QBS,   IF_REAL, "Bulk-Source charge storage"),
 OPU("qgs",   MOS9_QGS,            IF_REAL, "Gate-Source charge storage"),
 OPU("qgd",   MOS9_QGD,            IF_REAL, "Gate-Drain charge storage"),
 OPU("qgb",  MOS9_QGB,   IF_REAL, "Gate-Bulk charge storage"),
 OPU("qbd",  MOS9_QBD,   IF_REAL, "Bulk-Drain charge storage"),
 OPU("p",    MOS9_POWER, IF_REAL, "Instantaneous power"),
 OPU("sens_l_dc", MOS9_L_SENS_DC,    IF_REAL, "dc sensitivity wrt length"),
 OPU("sens_l_real",MOS9_L_SENS_REAL, IF_REAL, 
        "real part of ac sensitivity wrt length"),
 OPU("sens_l_imag",MOS9_L_SENS_IMAG, IF_REAL, 
        "imag part of ac sensitivity wrt length"),
 OPU("sens_l_cplx",MOS9_L_SENS_CPLX, IF_COMPLEX, "ac sensitivity wrt length"),
 OPU("sens_l_mag", MOS9_L_SENS_MAG,  IF_REAL, 
        "sensitivity wrt l of ac magnitude"),
 OPU("sens_l_ph",  MOS9_L_SENS_PH,   IF_REAL, "sensitivity wrt l of ac phase"),
 OPU("sens_w_dc",  MOS9_W_SENS_DC,   IF_REAL, "dc sensitivity wrt width"),
 OPU("sens_w_real",MOS9_W_SENS_REAL, IF_REAL, 
        "real part of ac sensitivity wrt width"),
 OPU("sens_w_imag",MOS9_W_SENS_IMAG, IF_REAL, 
        "imag part of ac sensitivity wrt width"),
 OPU("sens_w_mag", MOS9_W_SENS_MAG,  IF_REAL,
        "sensitivity wrt w of ac magnitude"),
 OPU("sens_w_ph",  MOS9_W_SENS_PH,   IF_REAL, "sensitivity wrt w of ac phase"),
 OPU("sens_w_cplx",MOS9_W_SENS_CPLX, IF_COMPLEX, "ac sensitivity wrt width")
};

IFparm MOS9mPTable[] = { /* model parameters */
 OP("type",   MOS9_MOD_TYPE,   IF_STRING   ,"N-channel or P-channel MOS"),
 IP("nmos",   MOS9_MOD_NMOS,  IF_FLAG   ,"N type MOSfet model"),
 IP("pmos",   MOS9_MOD_PMOS,  IF_FLAG   ,"P type MOSfet model"),
 IOP("vto",   MOS9_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOPR("vt0",   MOS9_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOP("kp",    MOS9_MOD_KP,    IF_REAL   ,"Transconductance parameter"),
 IOP("gamma", MOS9_MOD_GAMMA, IF_REAL   ,"Bulk threshold parameter"),
 IOP("phi",   MOS9_MOD_PHI,   IF_REAL   ,"Surface potential"),
 IOP("rd",    MOS9_MOD_RD,    IF_REAL   ,"Drain ohmic resistance"),
 IOP("rs",    MOS9_MOD_RS,    IF_REAL   ,"Source ohmic resistance"),
 IOPA("cbd",   MOS9_MOD_CBD,   IF_REAL   ,"B-D junction capacitance"),
 IOPA("cbs",   MOS9_MOD_CBS,   IF_REAL   ,"B-S junction capacitance"),
 IOP("is",    MOS9_MOD_IS,    IF_REAL   ,"Bulk junction sat. current"),
 IOP("pb",    MOS9_MOD_PB,    IF_REAL   ,"Bulk junction potential"),
 IOPA("cgso",  MOS9_MOD_CGSO,  IF_REAL   ,"Gate-source overlap cap."),
 IOPA("cgdo",  MOS9_MOD_CGDO,  IF_REAL   ,"Gate-drain overlap cap."),
 IOPA("cgbo",  MOS9_MOD_CGBO,  IF_REAL   ,"Gate-bulk overlap cap."),
 IOP("rsh",   MOS9_MOD_RSH,   IF_REAL   ,"Sheet resistance"),
 IOPA("cj",    MOS9_MOD_CJ,    IF_REAL   ,"Bottom junction cap per area"),
 IOP("mj",    MOS9_MOD_MJ,    IF_REAL   ,"Bottom grading coefficient"),
 IOPA("cjsw",  MOS9_MOD_CJSW,  IF_REAL   ,"Side junction cap per area"),
 IOP("mjsw",  MOS9_MOD_MJSW,  IF_REAL   ,"Side grading coefficient"),
 IOPU("js",    MOS9_MOD_JS,    IF_REAL   ,"Bulk jct. sat. current density"),
 IOP("tox",   MOS9_MOD_TOX,   IF_REAL   ,"Oxide thickness"),
 IOP("ld",    MOS9_MOD_LD,    IF_REAL   ,"Lateral diffusion"),
 IOP("xl",    MOS9_MOD_XL,    IF_REAL   ,"Length mask adjustment"),
 IOP("wd",    MOS9_MOD_WD,    IF_REAL   ,"Width Narrowing (Diffusion)"),
 IOP("xw",    MOS9_MOD_XW,    IF_REAL   ,"Width mask adjustment"),
 IOPU("delvto",   MOS9_MOD_DELVTO,   IF_REAL   ,"Threshold voltage Adjust"),
 IOPUR("delvt0",  MOS9_MOD_DELVTO,   IF_REAL   ,"Threshold voltage Adjust"),
 IOP("u0",    MOS9_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOPR("uo",    MOS9_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOP("fc",    MOS9_MOD_FC,    IF_REAL   ,"Forward bias jct. fit parm."),
 IOP("nsub",  MOS9_MOD_NSUB,  IF_REAL   ,"Substrate doping"),
 IOP("tpg",   MOS9_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nss",   MOS9_MOD_NSS,   IF_REAL   ,"Surface state density"),
 IOP("vmax",  MOS9_MOD_VMAX,  IF_REAL   ,"Maximum carrier drift velocity"),
 IOP("xj",    MOS9_MOD_XJ,    IF_REAL   ,"Junction depth"),
 IOP("nfs",   MOS9_MOD_NFS,   IF_REAL   ,"Fast surface state density"),
 IOP("xd",    MOS9_MOD_XD,    IF_REAL ,"Depletion layer width"),
 IOP("alpha", MOS9_MOD_ALPHA, IF_REAL ,"Alpha"),
 IOP("eta",   MOS9_MOD_ETA,   IF_REAL ,"Vds dependence of threshold voltage"),
 IOP("delta", MOS9_MOD_DELTA, IF_REAL   ,"Width effect on threshold"),
 IOP("input_delta", MOS9_DELTA, IF_REAL ,""),
 IOP("theta", MOS9_MOD_THETA, IF_REAL ,"Vgs dependence on mobility"),
 IOP("kappa", MOS9_MOD_KAPPA, IF_REAL ,"Kappa"),
 IOPU("tnom",  MOS9_MOD_TNOM,  IF_REAL ,"Parameter measurement temperature"),
 IOP("kf",     MOS9_MOD_KF,    IF_REAL ,"Flicker noise coefficient"),
 IOP("af",     MOS9_MOD_AF,    IF_REAL ,"Flicker noise exponent")
};

char *MOS9names[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int	MOS9nSize = NUMELEMS(MOS9names);
int	MOS9pTSize = NUMELEMS(MOS9pTable);
int	MOS9mPTSize = NUMELEMS(MOS9mPTable);
int	MOS9iSize = sizeof(MOS9instance);
int	MOS9mSize = sizeof(MOS9model);
