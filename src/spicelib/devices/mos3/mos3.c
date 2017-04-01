/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mos3defs.h"
#include "ngspice/suffix.h"

IFparm MOS3pTable[] = { /* parameters */ 
 IOPU("m",         MOS3_M,       IF_REAL   , "Multiplier"),
 IOPU("l",         MOS3_L,       IF_REAL   , "Length"),
 IOPU("w",         MOS3_W,       IF_REAL   , "Width"),
 IOPU("ad",        MOS3_AD,      IF_REAL   , "Drain area"),
 IOPU("as",        MOS3_AS,      IF_REAL   , "Source area"),
 IOPU("pd",        MOS3_PD,      IF_REAL   , "Drain perimeter"),
 IOPU("ps",        MOS3_PS,      IF_REAL   , "Source perimeter"),
 OP("id",    MOS3_CD,            IF_REAL, "Drain current"),
 OPR("cd",    MOS3_CD,            IF_REAL, "Drain current"),
 OPU("ibd",   MOS3_CBD,           IF_REAL, "B-D junction current"),
 OPU("ibs",   MOS3_CBS,           IF_REAL, "B-S junction current"),
 OPU("is",   MOS3_CS,    IF_REAL, "Source current"),
 OPU("ig",   MOS3_CG,    IF_REAL, "Gate current"),
 OPU("ib",   MOS3_CB,    IF_REAL, "Bulk current"),
 OP("vgs",   MOS3_VGS,            IF_REAL, "Gate-Source voltage"),
 OP("vds",   MOS3_VDS,            IF_REAL, "Drain-Source voltage"),
 OP("vbs",   MOS3_VBS,            IF_REAL, "Bulk-Source voltage"),
 OPU("vbd",   MOS3_VBD,            IF_REAL, "Bulk-Drain voltage"),
 IOPU("nrd",       MOS3_NRD,     IF_REAL   , "Drain squares"),
 IOPU("nrs",       MOS3_NRS,     IF_REAL   , "Source squares"),
 IP("off",        MOS3_OFF,     IF_FLAG   , "Device initially off"),
 IOPAU("icvds",       MOS3_IC_VDS,  IF_REAL   , "Initial D-S voltage"),
 IOPAU("icvgs",       MOS3_IC_VGS,  IF_REAL   , "Initial G-S voltage"),
 IOPAU("icvbs",       MOS3_IC_VBS,  IF_REAL   , "Initial B-S voltage"),
 IOPU("ic",     MOS3_IC,      IF_REALVEC, "Vector of D-S, G-S, B-S voltages"),
 IOPU("temp",      MOS3_TEMP,    IF_REAL   , "Instance operating temperature"),
  IOPU("dtemp",      MOS3_DTEMP,    IF_REAL   , "Instance temperature difference"),
 IP("sens_l",  MOS3_L_SENS, IF_FLAG, "flag to request sensitivity WRT length"),
 IP("sens_w",  MOS3_W_SENS, IF_FLAG, "flag to request sensitivity WRT width"),
 OPU("dnode",     MOS3_DNODE,   IF_INTEGER, "Number of drain node"),
 OPU("gnode",     MOS3_GNODE,   IF_INTEGER, "Number of gate node"),
 OPU("snode",     MOS3_SNODE,   IF_INTEGER, "Number of source node"),
 OPU("bnode",     MOS3_BNODE,   IF_INTEGER, "Number of bulk node"),
 OPU("dnodeprime", MOS3_DNODEPRIME,IF_INTEGER,"Number of internal drain node"),
 OPU("snodeprime", MOS3_SNODEPRIME,IF_INTEGER,"Number of internal source node"),
 OP("von",               MOS3_VON,           IF_REAL,    "Turn-on voltage"),
 OP("vdsat",       MOS3_VDSAT,         IF_REAL, "Saturation drain voltage"),
 OPU("sourcevcrit", MOS3_SOURCEVCRIT,   IF_REAL, "Critical source voltage"),
 OPU("drainvcrit",  MOS3_DRAINVCRIT,    IF_REAL, "Critical drain voltage"),
 OP("rs", MOS3_SOURCERESIST, IF_REAL,  "Source resistance"),
 OPU("sourceconductance", MOS3_SOURCECONDUCT, IF_REAL,  "Source conductance"),
 OP("rd",  MOS3_DRAINRESIST,  IF_REAL,  "Drain resistance"),
 OPU("drainconductance",  MOS3_DRAINCONDUCT,  IF_REAL,  "Drain conductance"),
 OP("gm",    MOS3_GM,            IF_REAL, "Transconductance"),
 OP("gds",   MOS3_GDS,           IF_REAL, "Drain-Source conductance"),
 OP("gmb",  MOS3_GMBS,           IF_REAL, "Bulk-Source transconductance"),
 OPR("gmbs",  MOS3_GMBS,         IF_REAL, "Bulk-Source transconductance"),
 OPU("gbd",   MOS3_GBD,           IF_REAL, "Bulk-Drain conductance"),
 OPU("gbs",   MOS3_GBS,           IF_REAL, "Bulk-Source conductance"),

 OP("cbd", MOS3_CAPBD,         IF_REAL, "Bulk-Drain capacitance"),
 OP("cbs", MOS3_CAPBS,         IF_REAL, "Bulk-Source capacitance"),
 OP("cgs", MOS3_CAPGS,         IF_REAL, "Gate-Source capacitance"),
/* OPR("cgs",       MOS3_CGS,     IF_REAL   , "Gate-Source capacitance"),*/
 OP("cgd", MOS3_CAPGD,         IF_REAL, "Gate-Drain capacitance"),
/* OPR("cgd",       MOS3_CGD,     IF_REAL   , "Gate-Drain capacitance"),*/
 OP("cgb", MOS3_CAPGB,	       IF_REAL, "Gate-Bulk capacitance"),

 OPU("cqgs",MOS3_CQGS,IF_REAL,"Capacitance due to gate-source charge storage"),
 OPU("cqgd",MOS3_CQGD, IF_REAL,"Capacitance due to gate-drain charge storage"),
 OPU("cqgb",MOS3_CQGB,  IF_REAL,"Capacitance due to gate-bulk charge storage"),
 OPU("cqbd",MOS3_CQBD,IF_REAL,"Capacitance due to bulk-drain charge storage"),
 OPU("cqbs",MOS3_CQBS,IF_REAL,"Capacitance due to bulk-source charge storage"),

 OPU("cbd0",MOS3_CAPZEROBIASBD,IF_REAL,"Zero-Bias B-D junction capacitance"),
 OPU("cbdsw0",MOS3_CAPZEROBIASBDSW,IF_REAL,
					"Zero-Bias B-D sidewall capacitance"),
 OPU("cbs0",MOS3_CAPZEROBIASBS,IF_REAL,"Zero-Bias B-S junction capacitance"),
 OPU("cbssw0",MOS3_CAPZEROBIASBSSW,IF_REAL,
					"Zero-Bias B-S sidewall capacitance"),
 OPU("qbs",  MOS3_QBS,   IF_REAL, "Bulk-Source charge storage"),
 OPU("qgs",   MOS3_QGS,            IF_REAL, "Gate-Source charge storage"),
 OPU("qgd",   MOS3_QGD,            IF_REAL, "Gate-Drain charge storage"),
 OPU("qgb",  MOS3_QGB,   IF_REAL, "Gate-Bulk charge storage"),
 OPU("qbd",  MOS3_QBD,   IF_REAL, "Bulk-Drain charge storage"),
 OPU("p",    MOS3_POWER, IF_REAL, "Instantaneous power"),
 OPU("sens_l_dc", MOS3_L_SENS_DC,    IF_REAL, "dc sensitivity wrt length"),
 OPU("sens_l_real",MOS3_L_SENS_REAL, IF_REAL, 
        "real part of ac sensitivity wrt length"),
 OPU("sens_l_imag",MOS3_L_SENS_IMAG, IF_REAL, 
        "imag part of ac sensitivity wrt length"),
 OPU("sens_l_cplx",MOS3_L_SENS_CPLX, IF_COMPLEX, "ac sensitivity wrt length"),
 OPU("sens_l_mag", MOS3_L_SENS_MAG,  IF_REAL, 
        "sensitivity wrt l of ac magnitude"),
 OPU("sens_l_ph",  MOS3_L_SENS_PH,   IF_REAL, "sensitivity wrt l of ac phase"),
 OPU("sens_w_dc",  MOS3_W_SENS_DC,   IF_REAL, "dc sensitivity wrt width"),
 OPU("sens_w_real",MOS3_W_SENS_REAL, IF_REAL, 
        "real part of ac sensitivity wrt width"),
 OPU("sens_w_imag",MOS3_W_SENS_IMAG, IF_REAL, 
        "imag part of ac sensitivity wrt width"),
 OPU("sens_w_mag", MOS3_W_SENS_MAG,  IF_REAL,
        "sensitivity wrt w of ac magnitude"),
 OPU("sens_w_ph",  MOS3_W_SENS_PH,   IF_REAL, "sensitivity wrt w of ac phase"),
 OPU("sens_w_cplx",MOS3_W_SENS_CPLX, IF_COMPLEX, "ac sensitivity wrt width")
};

IFparm MOS3mPTable[] = { /* model parameters */
 OP("type",   MOS3_MOD_TYPE,   IF_STRING   ,"N-channel or P-channel MOS"),
 IP("nmos",   MOS3_MOD_NMOS,  IF_FLAG   ,"N type MOSfet model"),
 IP("pmos",   MOS3_MOD_PMOS,  IF_FLAG   ,"P type MOSfet model"),
 IOP("vto",   MOS3_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOPR("vt0",   MOS3_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOP("kp",    MOS3_MOD_KP,    IF_REAL   ,"Transconductance parameter"),
 IOP("gamma", MOS3_MOD_GAMMA, IF_REAL   ,"Bulk threshold parameter"),
 IOP("phi",   MOS3_MOD_PHI,   IF_REAL   ,"Surface potential"),
 IOP("rd",    MOS3_MOD_RD,    IF_REAL   ,"Drain ohmic resistance"),
 IOP("rs",    MOS3_MOD_RS,    IF_REAL   ,"Source ohmic resistance"),
 IOPA("cbd",   MOS3_MOD_CBD,   IF_REAL   ,"B-D junction capacitance"),
 IOPA("cbs",   MOS3_MOD_CBS,   IF_REAL   ,"B-S junction capacitance"),
 IOP("is",    MOS3_MOD_IS,    IF_REAL   ,"Bulk junction sat. current"),
 IOP("pb",    MOS3_MOD_PB,    IF_REAL   ,"Bulk junction potential"),
 IOPA("cgso",  MOS3_MOD_CGSO,  IF_REAL   ,"Gate-source overlap cap."),
 IOPA("cgdo",  MOS3_MOD_CGDO,  IF_REAL   ,"Gate-drain overlap cap."),
 IOPA("cgbo",  MOS3_MOD_CGBO,  IF_REAL   ,"Gate-bulk overlap cap."),
 IOP("rsh",   MOS3_MOD_RSH,   IF_REAL   ,"Sheet resistance"),
 IOPA("cj",    MOS3_MOD_CJ,    IF_REAL   ,"Bottom junction cap per area"),
 IOP("mj",    MOS3_MOD_MJ,    IF_REAL   ,"Bottom grading coefficient"),
 IOPA("cjsw",  MOS3_MOD_CJSW,  IF_REAL   ,"Side junction cap per area"),
 IOP("mjsw",  MOS3_MOD_MJSW,  IF_REAL   ,"Side grading coefficient"),
 IOPU("js",    MOS3_MOD_JS,    IF_REAL   ,"Bulk jct. sat. current density"),
 IOP("tox",   MOS3_MOD_TOX,   IF_REAL   ,"Oxide thickness"),
 IOP("ld",    MOS3_MOD_LD,    IF_REAL   ,"Lateral diffusion"),
  IOP("xl",    MOS3_MOD_XL,    IF_REAL   ,"Length mask adjustment"),
 IOP("wd",    MOS3_MOD_WD,    IF_REAL   ,"Width Narrowing (Diffusion)"),
 IOP("xw",    MOS3_MOD_XW,    IF_REAL   ,"Width mask adjustment"),
 IOPU("delvto",   MOS3_MOD_DELVTO,   IF_REAL   ,"Threshold voltage Adjust"),
 IOPUR("delvt0",  MOS3_MOD_DELVTO,   IF_REAL   ,"Threshold voltage Adjust"),
 IOP("u0",    MOS3_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOPR("uo",    MOS3_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOP("fc",    MOS3_MOD_FC,    IF_REAL   ,"Forward bias jct. fit parm."),
 IOP("nsub",  MOS3_MOD_NSUB,  IF_REAL   ,"Substrate doping"),
 IOP("tpg",   MOS3_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nss",   MOS3_MOD_NSS,   IF_REAL   ,"Surface state density"),
 IOP("vmax",  MOS3_MOD_VMAX,  IF_REAL   ,"Maximum carrier drift velocity"),
 IOP("xj",    MOS3_MOD_XJ,    IF_REAL   ,"Junction depth"),
 IOP("nfs",   MOS3_MOD_NFS,   IF_REAL   ,"Fast surface state density"),
 IOP("xd",    MOS3_MOD_XD,    IF_REAL ,"Depletion layer width"),
 IOP("alpha", MOS3_MOD_ALPHA, IF_REAL ,"Alpha"),
 IOP("eta",   MOS3_MOD_ETA,   IF_REAL ,"Vds dependence of threshold voltage"),
 IOP("delta", MOS3_MOD_DELTA, IF_REAL   ,"Width effect on threshold"),
 IOP("input_delta", MOS3_DELTA, IF_REAL ,""),
 IOP("theta", MOS3_MOD_THETA, IF_REAL ,"Vgs dependence on mobility"),
 IOP("kappa", MOS3_MOD_KAPPA, IF_REAL ,"Kappa"),
 IOPU("tnom",  MOS3_MOD_TNOM,  IF_REAL ,"Parameter measurement temperature"),
 IOP("kf",     MOS3_MOD_KF,    IF_REAL ,"Flicker noise coefficient"),
 IOP("af",     MOS3_MOD_AF,    IF_REAL ,"Flicker noise exponent")
};

char *MOS3names[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int	MOS3nSize = NUMELEMS(MOS3names);
int	MOS3pTSize = NUMELEMS(MOS3pTable);
int	MOS3mPTSize = NUMELEMS(MOS3mPTable);
int	MOS3iSize = sizeof(MOS3instance);
int	MOS3mSize = sizeof(MOS3model);
