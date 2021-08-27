/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS: 2018 Holger Vogt, 2020 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "vdmosdefs.h"
#include "ngspice/suffix.h"

IFparm VDMOSpTable[] = { /* parameters */ 
 IOPU("m",          VDMOS_M,          IF_REAL,    "Multiplier"),
 IP("off",          VDMOS_OFF,        IF_FLAG,    "Device initially off"),
 IOPU("icvds",      VDMOS_IC_VDS,     IF_REAL,    "Initial D-S voltage"),
 IOPU("icvgs",      VDMOS_IC_VGS,     IF_REAL,    "Initial G-S voltage"),
 IOPU("temp",       VDMOS_TEMP,       IF_REAL,    "Instance temperature"),
 IOPU("dtemp",      VDMOS_DTEMP,      IF_REAL,    "Instance temperature difference"),
 IP( "ic",          VDMOS_IC,         IF_REALVEC, "Vector of D-S, G-S voltages"),

 IOP("thermal",     VDMOS_THERMAL,    IF_FLAG,    "Thermal model switch on/off"),

 OP( "id",          VDMOS_CD,         IF_REAL,    "Drain current"),
 OP( "is",          VDMOS_CS,         IF_REAL,    "Source current"),
 OP( "ig",          VDMOS_CG,         IF_REAL,    "Gate current"),
 OP( "vgs",         VDMOS_VGS,        IF_REAL,    "Gate-Source voltage"),
 OP( "vds",         VDMOS_VDS,        IF_REAL,    "Drain-Source voltage"),
 OP( "cgs",         VDMOS_CAPGS,      IF_REAL,    "Gate-Source capacitance"),
 OP( "cgd",         VDMOS_CAPGD,      IF_REAL,    "Gate-Drain capacitance"),
 OP( "cds",         VDMOS_CAPDS,      IF_REAL,    "Drain-Source capacitance"),
 OP( "idio",        VDMOS_CDIO,       IF_REAL,    "Body diode current"),

 OPU( "dnode",      VDMOS_DNODE,      IF_INTEGER, "Number of the drain node "),
 OPU( "gnode",      VDMOS_GNODE,      IF_INTEGER, "Number of the gate node "),
 OPU( "snode",      VDMOS_SNODE,      IF_INTEGER, "Number of the source node "),
 OPU( "tempnode",   VDMOS_TNODE,      IF_INTEGER, "Number of temperature node"),
 OPU( "tcasenode",  VDMOS_TCASE,      IF_INTEGER, "Number of 2nd temperature node"),
 OPU( "dnodeprime", VDMOS_DNODEPRIME, IF_INTEGER, "Number of int. drain node"),
 OPU( "snodeprime", VDMOS_SNODEPRIME, IF_INTEGER, "Number of int. source node "),

 OP( "von",               VDMOS_VON,           IF_REAL, "Device on state voltage"),
 OP( "rs",                VDMOS_SOURCERESIST,  IF_REAL, "Source resistance"),
 OPU("sourceconductance", VDMOS_SOURCECONDUCT, IF_REAL, "Conductance of source"),
 OP( "rd",                VDMOS_DRAINRESIST,   IF_REAL, "Drain conductance"),
 OPU("drainconductance",  VDMOS_DRAINCONDUCT,  IF_REAL, "Conductance of drain"),

 OP( "gm",        VDMOS_GM,         IF_REAL,    "Transconductance"),
 OP( "gds",       VDMOS_GDS,        IF_REAL,    "Drain-Source conductance"),

 OPU( "cqgs", VDMOS_CQGS, IF_REAL, "Capacitance due to gate-source charge storage"),
 OPU( "cqgd", VDMOS_CQGD, IF_REAL, "Capacitance due to gate-drain charge storage"),

 OPU( "qgs",      VDMOS_QGS,        IF_REAL,    "Gate-Source charge storage"),
 OPU( "qgd",      VDMOS_QGD,        IF_REAL,    "Gate-Drain charge storage"),
 OPU( "p",        VDMOS_POWER,      IF_REAL,    "Instantaneous power"),
};

IFparm VDMOSmPTable[] = { /* model parameters */
 /* basic device */
 OP("type",   VDMOS_MOD_TYPE,  IF_STRING, "N-channel or P-channel MOS"),
 IOP("vto",   VDMOS_MOD_VTH,   IF_REAL,   "Threshold voltage"),
 IOPR("vth0", VDMOS_MOD_VTH,   IF_REAL,   "Threshold voltage"),
 IOP("kp",    VDMOS_MOD_KP,    IF_REAL,   "Transconductance parameter"),
 IOP("phi",   VDMOS_MOD_PHI,   IF_REAL,   "Surface potential"),
 IOP("lambda",VDMOS_MOD_LAMBDA,IF_REAL,   "Channel length modulation"),
 IOP("theta", VDMOS_MOD_THETA, IF_REAL,   "Vgs dependence on mobility"),
 IOP("rd",    VDMOS_MOD_RD,    IF_REAL,   "Drain ohmic resistance"),
 IOP("rs",    VDMOS_MOD_RS,    IF_REAL,   "Source ohmic resistance"),
 IOP("rg",    VDMOS_MOD_RG,    IF_REAL,   "Gate ohmic resistance"),
 IOP("tnom",  VDMOS_MOD_TNOM,  IF_REAL,   "Parameter measurement temperature"),
 IOP("kf",    VDMOS_MOD_KF,    IF_REAL,   "Flicker noise coefficient"),
 IOP("af",    VDMOS_MOD_AF,    IF_REAL,   "Flicker noise exponent"),
 IP("vdmosn", VDMOS_MOD_NMOS,  IF_FLAG,   "N type DMOSfet model"),
 IP("vdmosp", VDMOS_MOD_PMOS,  IF_FLAG,   "P type DMOSfet model"),
 IP("vdmos",  VDMOS_MOD_DMOS,  IF_REAL,   "DMOS transistor"),

 /* quasi saturation */
 IOP("rq",      VDMOS_MOD_RQ,      IF_REAL,   "Quasi saturation resistance fitting parameter"),
 IOP("vq",      VDMOS_MOD_VQ,      IF_REAL,   "Quasi saturation voltage fitting parameter"),
 IOP("mtriode", VDMOS_MOD_MTRIODE, IF_REAL,   "Conductance multiplier in triode region"),

 /* temperature dependency */
 IOP( "tcvth", VDMOS_MOD_TCVTH, IF_REAL,  "Linear Vth0 temperature coefficient"),
 IOPR("vtotc", VDMOS_MOD_TCVTH, IF_REAL,  "Linear Vth0 temperature coefficient"),
 IOP( "mu",    VDMOS_MOD_MU,    IF_REAL,  "Exponent of gain temperature dependency"),
 IOPR("bex",   VDMOS_MOD_MU,    IF_REAL,  "Exponent of gain temperature dependency"),
 IOP( "texp0", VDMOS_MOD_TEXP0, IF_REAL,  "Drain resistance rd0 temperature exponent"),
 IOP( "texp1", VDMOS_MOD_TEXP1, IF_REAL,  "Drain resistance rd1 temperature exponent"),
 IOP( "trd1",  VDMOS_MOD_TRD1,  IF_REAL,  "Drain resistance linear temperature coefficient"),
 IOP( "trd2",  VDMOS_MOD_TRD2,  IF_REAL,  "Drain resistance quadratic temperature coefficient"),
 IOP( "trg1",  VDMOS_MOD_TRG1,  IF_REAL,  "Gate resistance linear temperature coefficient"),
 IOP( "trg2",  VDMOS_MOD_TRG2,  IF_REAL,  "Gate resistance quadratic temperature coefficient"),
 IOP( "trs1",  VDMOS_MOD_TRS1,  IF_REAL,  "Source resistance linear temperature coefficient"),
 IOP( "trs2",  VDMOS_MOD_TRS2,  IF_REAL,  "Source resistance quadratic temperature coefficient"),
 IOP( "trb1",  VDIO_MOD_TRB1,  IF_REAL,  "Body resistance linear temperature coefficient"),
 IOP( "trb2",  VDIO_MOD_TRB2,  IF_REAL,  "Body resistance quadratic temperature coefficient"),

 /* weak inversion */
 IOP("subshift",  VDMOS_MOD_SUBSHIFT,   IF_REAL, "Shift of weak inversion plot on the vgs axis"),
 IOP("ksubthres", VDMOS_MOD_KSUBTHRES,  IF_REAL, "Slope of weak inversion log current versus vgs"),
 IOP("tksubthres1", VDMOS_MOD_TKSUBTHRES1, IF_REAL,  "Linear temperature coefficient of ksubthres"),
 IOP("tksubthres2", VDMOS_MOD_TKSUBTHRES2, IF_REAL,  "Quadratic temperature coefficient of ksubthres"),

/* body diode */
 IOP("bv",    VDIO_MOD_BV,    IF_REAL,   "Vds breakdown voltage"),
 IOP("ibv",   VDIO_MOD_IBV,   IF_REAL,   "Current at Vds=bv"),
 IOP("nbv",   VDIO_MOD_NBV,   IF_REAL,   "Vds breakdown emission coefficient"),
 IOP("rds",   VDMOS_MOD_RDS,   IF_REAL,   "Drain-source shunt resistance"),
 IOP("rb",    VDIO_MOD_RB,    IF_REAL,   "Body diode ohmic resistance"),
 IOP("n",     VDIO_MOD_N,     IF_REAL,   "Body diode emission coefficient"),
 IOP("tt",    VDIO_MOD_TT,    IF_REAL,   "Body diode transit time"),
 IOP("eg",    VDIO_MOD_EG,    IF_REAL,   "Body diode activation energy for temperature effect on Is"),
 IOP("xti",   VDIO_MOD_XTI,   IF_REAL,   "Body diode saturation current temperature exponent"),
 IOP("is",    VDIO_MOD_IS,    IF_REAL,   "Body diode saturation current"),
 IOP("vj",    VDIO_MOD_VJ,    IF_REAL,   "Body diode junction potential"),

 /* body diode capacitance (e.g. source-drain capacitance) */
 IOPA("cjo",  VDIO_MOD_CJ,    IF_REAL,   "Zero-bias body diode junction capacitance"),
 IOP("m",     VDIO_MOD_MJ,    IF_REAL,   "Body diode grading coefficient"),
 IOP("fc",    VDIO_MOD_FC,    IF_REAL,   "Body diode coefficient for forward-bias depletion capacitance formula"),

 /* gate-source and gate-drain capacitances */
 IOPA("cgdmin", VDMOS_MOD_CGDMIN, IF_REAL, "Minimum non-linear G-D capacitance"),
 IOPA("cgdmax", VDMOS_MOD_CGDMAX, IF_REAL, "Maximum non-linear G-D capacitance"),
 IOPA("a",      VDMOS_MOD_A,      IF_REAL, "Non-linear Cgd capacitance parameter"),
 IOPA("cgs",    VDMOS_MOD_CGS,    IF_REAL, "Gate-source capacitance"),

/* self heating */
 IOP("rthjc",  VDMOS_MOD_RTHJC,  IF_REAL, "Self-heating thermal resistance, junction-to-case"),
 IOP("rthca",  VDMOS_MOD_RTHCA,  IF_REAL, "Self-heating thermal resistance, case-to-ambient"),
 IOP("cthj",   VDMOS_MOD_CTHJ,  IF_REAL, "Self-heating thermal capacitance"),
 
/* soa check */
 IOP("vgs_max", VDMOS_MOD_VGS_MAX, IF_REAL, "maximum voltage G-S branch"),
 IOP("vgd_max", VDMOS_MOD_VGD_MAX, IF_REAL, "maximum voltage G-D branch"),
 IOP("vds_max", VDMOS_MOD_VDS_MAX, IF_REAL, "maximum voltage D-S branch"),
 IOP("vgsr_max", VDMOS_MOD_VGSR_MAX, IF_REAL, "maximum voltage G-S branch"),
 IOP("vgdr_max", VDMOS_MOD_VGDR_MAX, IF_REAL, "maximum voltage G-D branch"),
 IOP("pd_max", VDMOS_MOD_PD_MAX, IF_REAL, "maximum device power dissipation"),
 IOP("id_max", VDMOS_MOD_ID_MAX, IF_REAL, "maximum drain/source current"),
 IOP("idr_max", VDMOS_MOD_IDR_MAX, IF_REAL, "maximum drain/source reverse current"),
 IOP("te_max", VDMOS_MOD_TE_MAX, IF_REAL, "maximum temperature"),
 IOP("rth_ext", VDMOS_MOD_RTH_EXT, IF_REAL, "thermal resistance case to ambient, incl. heat sink"),
 IOP("derating", VDMOS_MOD_DERATING, IF_REAL, "thermal derating for power"),
};

char *VDMOSnames[] = {
    "Drain",
    "Gate",
    "Source",
    "Temp",
    "Tcase"
};

int VDMOSnSize = NUMELEMS(VDMOSnames);
int VDMOSpTSize = NUMELEMS(VDMOSpTable);
int VDMOSmPTSize = NUMELEMS(VDMOSmPTable);
int VDMOSiSize = sizeof(VDMOSinstance);
int VDMOSmSize = sizeof(VDMOSmodel);
