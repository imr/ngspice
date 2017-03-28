/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "mesadefs.h"
#include "ngspice/suffix.h"


IFparm MESApTable[] = { /* parameters */ 
 OP("off",       MESA_OFF,            IF_FLAG   ,"Device initially off"),
 IOP("l",        MESA_LENGTH,         IF_REAL   ,"Length of device"),
 IOP("w",        MESA_WIDTH,          IF_REAL   ,"Width of device"),
 IOP("m",        MESA_M,              IF_REAL   ,"Parallel Multiplier"),
 IOP("icvds",    MESA_IC_VDS,         IF_REAL   ,"Initial D-S voltage"),
 IOP("icvgs",    MESA_IC_VGS,         IF_REAL   ,"Initial G-S voltage"),
 IOP("td",       MESA_TD,             IF_REAL   ,"Instance drain temperature"),
 IOP("ts",       MESA_TS,             IF_REAL   ,"Instance source temperature"),
 IOP("dtemp",    MESA_DTEMP,          IF_REAL   ,"Instance temperature difference"),
 OP("dnode",     MESA_DRAINNODE,      IF_INTEGER,"Number of drain node"),
 OP("gnode",     MESA_GATENODE,       IF_INTEGER,"Number of gate node"),
 OP("snode",     MESA_SOURCENODE,     IF_INTEGER,"Number of source node"),
 OP("dprimenode",MESA_DRAINPRIMENODE, IF_INTEGER,"Number of internal drain node"),
 OP("sprimenode",MESA_SOURCEPRIMENODE,IF_INTEGER,"Number of internal source node"),
 OP("gprimenode",MESA_GATEPRIMENODE,  IF_INTEGER,"Number of internal gate node"),
 OP("vgs",       MESA_VGS,            IF_REAL,"Gate-Source voltage"),
 OP("vgd",       MESA_VGD,            IF_REAL,"Gate-Drain voltage"),
 OP("cg",        MESA_CG,             IF_REAL,"Gate capacitance"),
 OP("cd",        MESA_CD,             IF_REAL,"Drain capacitance"),
 OP("cgd",       MESA_CGD,            IF_REAL,"Gate_Drain capacitance"),
 OP("gm",        MESA_GM,             IF_REAL,"Transconductance"),
 OP("gds",       MESA_GDS,            IF_REAL,"Drain-Source conductance"),
 OP("ggs",       MESA_GGS,            IF_REAL,"Gate-Source conductance"),
 OP("ggd",       MESA_GGD,            IF_REAL,"Gate-Drain conductance"),
 OP("qgs",       MESA_QGS,            IF_REAL,"Gate-Source charge storage"),
 OP("cqgs",      MESA_CQGS,           IF_REAL,"Capacitance due to gate-source charge storage"),
 OP("qgd",       MESA_QGD,            IF_REAL,"Gate-Drain charge storage"),
 OP("cqgd",      MESA_CQGD,           IF_REAL,"Capacitance due to gate-drain charge storage"),
 OP("cs",        MESA_CS,             IF_REAL,"Source current"),
 OP("p",         MESA_POWER,          IF_REAL,"Power dissipated by the mesfet")

};

IFparm MESAmPTable[] = { /* model parameters */
 OP( "type",    MESA_MOD_TYPE,   IF_STRING,"N-type or P-type MESfet model"),
 IOP( "vto",    MESA_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOPR("vt0",    MESA_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOP( "lambda", MESA_MOD_LAMBDA, IF_REAL,"Output conductance parameter"),
 IOP( "lambdahf",MESA_MOD_LAMBDAHF, IF_REAL,"Output conductance parameter at high frequencies"),
 IOP( "beta",   MESA_MOD_BETA,   IF_REAL,"Transconductance parameter"),
 IOP( "vs",     MESA_MOD_VS,     IF_REAL,"Saturation velocity"),
 IOP( "rd",     MESA_MOD_RD,     IF_REAL,"Drain ohmic resistance"),
 IOP( "rs",     MESA_MOD_RS,     IF_REAL,"Source ohmic resistance"),
 IOP( "rg",     MESA_MOD_RG,     IF_REAL,"Gate ohmic resistance"),
 IOP( "ri",     MESA_MOD_RI,     IF_REAL,"Gate-source ohmic resistance"),
 IOP( "rf",     MESA_MOD_RF,     IF_REAL,"Gate-drain ohmic resistance"),
 IOP( "rdi",    MESA_MOD_RDI,    IF_REAL,"Intrinsic source ohmic resistance"),
 IOP( "rsi",    MESA_MOD_RSI,    IF_REAL,"Intrinsic drain ohmic resistance"),
 IOP( "phib",   MESA_MOD_PHIB,   IF_REAL,"Effective Schottky barrier height at room temperature"),
 IOP( "phib1",  MESA_MOD_PHIB1,  IF_REAL,""),
 IOPR("tphib",  MESA_MOD_PHIB1,  IF_REAL,""),
 IOP( "astar",  MESA_MOD_ASTAR,  IF_REAL,"Effective Richardson constant"),
 IOP( "ggr",    MESA_MOD_GGR,    IF_REAL,"Reverse diode conductance"),
 IOP( "del",    MESA_MOD_DEL,    IF_REAL,""),
 IOP( "xchi",   MESA_MOD_XCHI,   IF_REAL,""),
 IOPR("tggr",   MESA_MOD_XCHI,   IF_REAL,""),
 IOP( "n",      MESA_MOD_N,      IF_REAL,"Emission coefficient"),
 IOP( "eta",    MESA_MOD_ETA,    IF_REAL,"Subthreshold ideality factor"),
 IOP( "m",      MESA_MOD_M,      IF_REAL,"Knee shape parameter"),
 IOP( "mc",     MESA_MOD_MC,     IF_REAL,"Knee shape parameter"),
 IOP( "alpha",  MESA_MOD_ALPHA,  IF_REAL,""),
 IOP( "sigma0", MESA_MOD_SIGMA0, IF_REAL,"Threshold voltage coefficient"),
 IOP( "vsigmat",MESA_MOD_VSIGMAT,IF_REAL,""),
 IOP( "vsigma", MESA_MOD_VSIGMA, IF_REAL,""),
 IOP( "mu",     MESA_MOD_MU,     IF_REAL,"Mobility"),
 IOP( "theta",  MESA_MOD_THETA,  IF_REAL,""),
 IOP( "mu1",    MESA_MOD_MU1,    IF_REAL,"Second moblity parameter"),
 IOP( "mu2",    MESA_MOD_MU2,    IF_REAL,"Third moblity parameter"),
 IOP( "d",      MESA_MOD_D,      IF_REAL,"Depth of device"),
 IOP( "nd",     MESA_MOD_ND,     IF_REAL,"Doping density"),
 IOP( "du",     MESA_MOD_DU,     IF_REAL,"Depth of device"),
 IOP( "ndu",    MESA_MOD_NDU,    IF_REAL,"Doping density"),
 IOP( "th",     MESA_MOD_TH,     IF_REAL,"Thickness of delta doped layer"),
 IOP( "ndelta", MESA_MOD_NDELTA, IF_REAL,"Delta doped layer doping density"),
 IOP( "delta",  MESA_MOD_DELTA,  IF_REAL,""),
 IOP( "tc",     MESA_MOD_TC,     IF_REAL,"Transconductance compression factor"),
 IOP( "tvto",   MESA_MOD_TVTO,   IF_REAL,"Temperature coefficient for vto"),
 IOPR("alphat", MESA_MOD_TVTO,   IF_REAL,""),
 IOP( "tlambda",MESA_MOD_TLAMBDA,IF_REAL,"Temperature coefficient for lambda"),
 IOP( "teta0",  MESA_MOD_TETA0,  IF_REAL,"First temperature coefficient for eta"),
 IOP( "teta1",  MESA_MOD_TETA1,  IF_REAL,"Second temperature coefficient for eta"),
 IOP( "tmu",    MESA_MOD_TMU,    IF_REAL,"Temperature coefficient for mobility"),
 IOP( "xtm0",   MESA_MOD_XTM0,   IF_REAL,"First exponent for temp dependence of mobility"),
 IOP( "xtm1",   MESA_MOD_XTM1,   IF_REAL,"Second exponent for temp dependence of mobility"),
 IOP( "xtm2",   MESA_MOD_XTM2,   IF_REAL,"Third exponent for temp dependence of mobility"),
 IOP( "ks",     MESA_MOD_KS,     IF_REAL,"Sidegating coefficient"),
 IOP( "vsg",    MESA_MOD_VSG,    IF_REAL,"Sidegating voltage"),
 IOP( "tf",     MESA_MOD_TF,     IF_REAL,"Characteristic temperature determined by traps"),
 IOP( "flo",    MESA_MOD_FLO,    IF_REAL,""),
 IOP( "delfo",  MESA_MOD_DELFO,  IF_REAL,""),
 IOP( "ag",     MESA_MOD_AG,     IF_REAL,""),
 IOP( "rtc1",   MESA_MOD_TC1,    IF_REAL,""),
 IOP( "rtc2",   MESA_MOD_TC2,    IF_REAL,""),
 IOP( "zeta",   MESA_MOD_ZETA,   IF_REAL,""),
 IOP( "level",  MESA_MOD_LEVEL,  IF_REAL,""),
 IOP( "nmax",   MESA_MOD_NMAX,   IF_REAL,""),
 IOP( "gamma",  MESA_MOD_GAMMA,  IF_REAL,""),
 IOP( "epsi",   MESA_MOD_EPSI,   IF_REAL,""), 
 IOP( "cas",    MESA_MOD_CAS,    IF_REAL,""),
 IOP( "cbs",    MESA_MOD_CBS,    IF_REAL,""),
 IP( "pmf",	    MESA_MOD_PMF,    IF_FLAG,"P type MESfet model"), 
 IP( "nmf",     MESA_MOD_NMF,    IF_FLAG,"N type MESfet model"),
 OP( "gd",      MESA_MOD_DRAINCONDUCT,   IF_REAL,"Drain conductance"),
 OP( "gs",      MESA_MOD_SOURCECONDUCT,  IF_REAL,"Source conductance"),
 OP( "vcrit",   MESA_MOD_VCRIT,  IF_REAL,"Critical voltage"),
};

char *MESAnames[] = {
    "Drain",
    "Gate",
    "Source"
};

int MESAnSize = NUMELEMS(MESAnames);
int MESApTSize = NUMELEMS(MESApTable);
int MESAmPTSize = NUMELEMS(MESAmPTable);
int MESAiSize = sizeof(MESAinstance);
int MESAmSize = sizeof(MESAmodel);
