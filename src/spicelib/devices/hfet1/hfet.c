/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "hfetdefs.h"
#include "ngspice/suffix.h"


IFparm HFETApTable[] = { /* parameters */ 
 OP("off",       HFETA_OFF,            IF_FLAG   ,"Device initially off"),
 IOP("l",        HFETA_LENGTH,         IF_REAL   ,"Length of device"),
 IOP("w",        HFETA_WIDTH,          IF_REAL   ,"Width of device"),
 IOP("m",        HFETA_M,              IF_REAL   ,"Parallel Multiplier"),
 IOP("icvds",    HFETA_IC_VDS,         IF_REAL   ,"Initial D-S voltage"),
 IOP("icvgs",    HFETA_IC_VGS,         IF_REAL   ,"Initial G-S voltage"),
 IOP("temp",     HFETA_TEMP,           IF_REAL   ,"Instance temperature"),
 IOP("dtemp",    HFETA_DTEMP,          IF_REAL   ,"Instance temperature difference"),
 OP("dnode",     HFETA_DRAINNODE,      IF_INTEGER,"Number of drain node"),
 OP("gnode",     HFETA_GATENODE,       IF_INTEGER,"Number of gate node"),
 OP("snode",     HFETA_SOURCENODE,     IF_INTEGER,"Number of source node"),
 OP("dprimenode",HFETA_DRAINPRIMENODE, IF_INTEGER,"Number of internal drain node"),
 OP("sprimenode",HFETA_SOURCEPRIMENODE,IF_INTEGER,"Number of internal source node"),
 OP("vgs",       HFETA_VGS,            IF_REAL,"Gate-Source voltage"),
 OP("vgd",       HFETA_VGD,            IF_REAL,"Gate-Drain voltage"),
 OP("cg",        HFETA_CG,             IF_REAL,"Gate capacitance"),
 OP("cd",        HFETA_CD,             IF_REAL,"Drain capacitance"),
 OP("cgd",       HFETA_CGD,            IF_REAL,"Gate_Drain capacitance"),
 OP("gm",        HFETA_GM,             IF_REAL,"Transconductance"),
 OP("gds",       HFETA_GDS,            IF_REAL,"Drain-Source conductance"),
 OP("ggs",       HFETA_GGS,            IF_REAL,"Gate-Source conductance"),
 OP("ggd",       HFETA_GGD,            IF_REAL,"Gate-Drain conductance"),
 OP("qgs",       HFETA_QGS,            IF_REAL,"Gate-Source charge storage"),
 OP("cqgs",      HFETA_CQGS,           IF_REAL,"Capacitance due to gate-source charge storage"),
 OP("qgd",       HFETA_QGD,            IF_REAL,"Gate-Drain charge storage"),
 OP("cqgd",      HFETA_CQGD,           IF_REAL,"Capacitance due to gate-drain charge storage"),
 OP("cs",        HFETA_CS,             IF_REAL   ,"Source current"),
 OP("p",         HFETA_POWER,          IF_REAL   ,"Power dissipated by the mesfet")

};

IFparm HFETAmPTable[] = { /* model parameters */
 IOP( "vt0",     HFETA_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOPR("vto",     HFETA_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOP( "lambda",  HFETA_MOD_LAMBDA, IF_REAL,"Output conductance parameter"),
 IOP( "rd",      HFETA_MOD_RD,     IF_REAL,"Drain ohmic resistance"),
 IOP( "rs",      HFETA_MOD_RS,     IF_REAL,"Source ohmic resistance"),
 IOP( "rg",      HFETA_MOD_RG,     IF_REAL,"Gate ohmic resistance"),
 IOP( "rdi",     HFETA_MOD_RDI,    IF_REAL,"Drain ohmic resistance"),
 IOP( "rsi",     HFETA_MOD_RSI,    IF_REAL,"Source ohmic resistance"),
 IOP( "rgs",     HFETA_MOD_RGS,    IF_REAL,"Gate-source ohmic resistance"),
 IOP( "rgd",     HFETA_MOD_RGD,    IF_REAL,"Gate-drain ohmic resistance"),
 IOP( "ri",      HFETA_MOD_RI,     IF_REAL,""),
 IOP( "rf",      HFETA_MOD_RF,     IF_REAL,""),
 IOP( "eta",     HFETA_MOD_ETA,    IF_REAL,"Subthreshold ideality factor"),
 IOP( "m",       HFETA_MOD_M,      IF_REAL,"Knee shape parameter"),
 IOP( "mc",      HFETA_MOD_MC,     IF_REAL,"Knee shape parameter"),
 IOP( "gamma",   HFETA_MOD_GAMMA,  IF_REAL,"Knee shape parameter"),
 IOP( "sigma0",  HFETA_MOD_SIGMA0, IF_REAL,"Threshold voltage coefficient"),
 IOP( "vsigmat", HFETA_MOD_VSIGMAT,IF_REAL,""),
 IOP( "vsigma",  HFETA_MOD_VSIGMA, IF_REAL,""),
 IOP( "mu",      HFETA_MOD_MU,     IF_REAL,"Moblity"),
 IOP( "di",      HFETA_MOD_DI,     IF_REAL,"Depth of device"),
 IOP( "delta",   HFETA_MOD_DELTA,  IF_REAL,""),
 IOP( "vs",      HFETA_MOD_VS,     IF_REAL,"Saturation velocity"),
 IOP( "nmax",    HFETA_MOD_NMAX,   IF_REAL,""),
 IOP( "deltad",  HFETA_MOD_DELTAD, IF_REAL,"Thickness correction"),
 IOP( "js1d",    HFETA_MOD_JS1D,   IF_REAL,""),
 IOP( "js2d",    HFETA_MOD_JS2D,   IF_REAL,""),
 IOP( "js1s",    HFETA_MOD_JS1S,   IF_REAL,""),
 IOP( "js2s",    HFETA_MOD_JS2S,   IF_REAL,""),
 IOP( "m1d",     HFETA_MOD_M1D,    IF_REAL,""),
 IOP( "m2d",     HFETA_MOD_M2D,    IF_REAL,""),
 IOP( "m1s",     HFETA_MOD_M1S,    IF_REAL,""),
 IOP( "m2s",     HFETA_MOD_M2S,    IF_REAL,""),
 IOP( "epsi",    HFETA_MOD_EPSI,   IF_REAL,""),
 IOP( "p",       HFETA_MOD_P,      IF_REAL,""),
 IOP( "cm3",     HFETA_MOD_CM3,    IF_REAL,""), 
 IOP( "a1",      HFETA_MOD_A1,     IF_REAL,""),
 IOP( "a2",      HFETA_MOD_A2,     IF_REAL,""),
 IOP( "mv1",     HFETA_MOD_MV1,    IF_REAL,""),
 IOP( "kappa",   HFETA_MOD_KAPPA,  IF_REAL,""),
 IOP( "delf",    HFETA_MOD_DELF,   IF_REAL,""),
 IOP( "fgds",    HFETA_MOD_FGDS,   IF_REAL,""),
 IOP( "tf",      HFETA_MOD_TF,     IF_REAL,""),
 IOP( "cds",     HFETA_MOD_CDS,    IF_REAL,""),
 IOP( "phib",    HFETA_MOD_PHIB,   IF_REAL,""),
 IOP( "talpha",  HFETA_MOD_TALPHA, IF_REAL,""),
 IOP( "mt1",     HFETA_MOD_MT1,    IF_REAL,""),
 IOP( "mt2",     HFETA_MOD_MT2,    IF_REAL,""),
 IOP( "ck1",     HFETA_MOD_CK1,    IF_REAL,""),
 IOP( "ck2",     HFETA_MOD_CK2,    IF_REAL,""),
 IOP( "cm1",     HFETA_MOD_CM1,    IF_REAL,""),
 IOP( "cm2",     HFETA_MOD_CM2,    IF_REAL,""),
 IOP( "astar",   HFETA_MOD_ASTAR,  IF_REAL,""),
 IOP( "eta1",    HFETA_MOD_ETA1,   IF_REAL,""),
 IOP( "d1",      HFETA_MOD_D1,     IF_REAL,""),
 IOP( "vt1",     HFETA_MOD_VT1,    IF_REAL,""),
 IOP( "eta2",    HFETA_MOD_ETA2,   IF_REAL,""),
 IOP( "d2",      HFETA_MOD_D2,     IF_REAL,""),
 IOP( "vt2",     HFETA_MOD_VT2,    IF_REAL,""),
 IOP( "ggr",     HFETA_MOD_GGR,    IF_REAL,""),
 IOP( "del",     HFETA_MOD_DEL,    IF_REAL,""),
 IOP( "gatemod", HFETA_MOD_GATEMOD,IF_INTEGER,""),
 IOP( "klambda", HFETA_MOD_KLAMBDA, IF_REAL,""),
 IOP( "kmu",     HFETA_MOD_KMU,     IF_REAL,""),
 IOP( "kvto",    HFETA_MOD_KVTO,    IF_REAL,""),
  OP( "type",	  HFETA_MOD_TYPE,    IF_STRING, "NHFET or PHFET"),
 IOP( "nhfet",   HFETA_MOD_NHFET,  IF_FLAG,"N HFET device"),
 IOP( "phfet",   HFETA_MOD_PHFET,  IF_FLAG,"P HFET device"),
};

char *HFETAnames[] = {
    "Drain",
    "Gate",
    "Source"
};

int HFETAnSize = NUMELEMS(HFETAnames);
int HFETApTSize = NUMELEMS(HFETApTable);
int HFETAmPTSize = NUMELEMS(HFETAmPTable);
int HFETAiSize = sizeof(HFETAinstance);
int HFETAmSize = sizeof(HFETAmodel);
