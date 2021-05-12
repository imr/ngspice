/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

/*
 * This file defines the HICUM data structures that are
 * available to the next level(s) up the calling hierarchy
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "hicum2defs.h"
#include "ngspice/suffix.h"

IFparm HICUMpTable[] = { /* parameters */
 IOPU("area",    HICUM_AREA,   IF_REAL,    "Area factor"),
 IOPU("off",     HICUM_OFF,    IF_FLAG,    "Device initially off"),
 IP("ic",        HICUM_IC,     IF_REALVEC, "Initial condition vector"),

// these are left here for future. Sometimes it is advantageous for debugging if one can set the initial node voltages of all nodes.
//  OP("icvb",  HICUM_IC_VB, IF_REAL,    "Initial B potential"),
//  OP("icvc",  HICUM_IC_VC, IF_REAL,    "Initial C potential"),
//  OP("icve",  HICUM_IC_VE, IF_REAL,    "Initial E potential"),
//  OP("icvbi", HICUM_IC_VBi, IF_REAL,    "Initial Bi potential"),
//  OP("icvbp", HICUM_IC_VBp, IF_REAL,    "Initial Bp potential"),
//  OP("icvci", HICUM_IC_VCi, IF_REAL,    "Initial Ci potential"),
//  OP("icvt", HICUM_IC_Vt, IF_REAL,    "Initial T potential"),
//  OP("icvei", HICUM_IC_VEi, IF_REAL,    "Initial Ei potential"),
 
 IOPU("m",       HICUM_M,      IF_REAL,    "Multiplier"),
 IOPU("temp",    HICUM_TEMP,   IF_REAL,    "Instance temperature"),
 IP("dt",        HICUM_DTEMP,  IF_REAL,    "Instance delta temperature"),
 IOPU("tk",     HICUM_QUEST_TK,     IF_REAL,    "Actual device temperature"),
 IOPU("dtsh",    HICUM_QUEST_DTSH,   IF_REAL,    "Temperature increase due to self-heating"),
 IOPU("it",    HICUM_QUEST_IT,   IF_REAL,    "transfer current"),

 OPU("collnode", HICUM_QUEST_COLLNODE, IF_INTEGER, "Number of collector node"),
 OPU("basenode", HICUM_QUEST_BASENODE, IF_INTEGER, "Number of base node"),
 OPU("emitnode", HICUM_QUEST_EMITNODE, IF_INTEGER, "Number of emitter node"),
 OPU("subsnode", HICUM_QUEST_SUBSNODE, IF_INTEGER, "Number of substrate node"),
 OPU("tempnode", HICUM_QUEST_TEMPNODE, IF_INTEGER, "Number of temperature node"),
 OPU("collCInode", HICUM_QUEST_COLLCINODE, IF_INTEGER, "Internal collector node"),
 OPU("baseBPnode", HICUM_QUEST_BASEBPNODE, IF_INTEGER, "External base node"),
 OPU("baseBInode", HICUM_QUEST_BASEBINODE, IF_INTEGER, "Internal base node"),
 OPU("emitEInode", HICUM_QUEST_EMITEINODE, IF_INTEGER, "Internal emitter node"),
 OPU("subsSInode", HICUM_QUEST_SUBSSINODE, IF_INTEGER, "Internal substrate node"),
 OPU("xfnode",  HICUM_QUEST_XFNODE,  IF_INTEGER, "Internal phase node xf"),
 OPU("xf1node", HICUM_QUEST_XF1NODE, IF_INTEGER, "Internal phase node xf1"),
 OPU("xf2node", HICUM_QUEST_XF2NODE, IF_INTEGER, "Internal phase node xf2"),
/* voltages */
 OP("vbe",    HICUM_QUEST_VBE,   IF_REAL, "External BE voltage"),
 OP("vbbp",   HICUM_QUEST_VBBP,  IF_REAL, "BBP voltage"),
 OP("vbc",    HICUM_QUEST_VBC,   IF_REAL, "External BC voltage"),
 OP("vce",    HICUM_QUEST_VCE,   IF_REAL, "External CE voltage"),
 OP("vsc",    HICUM_QUEST_VSC,   IF_REAL, "External SC voltage"),
 OP("vbiei",    HICUM_QUEST_VBIEI,   IF_REAL, "Internal BE voltage"),
 OP("vbpbi",    HICUM_QUEST_VBPBI,   IF_REAL, "Peripheral Base to internal Base voltage"),
 OP("vbici",    HICUM_QUEST_VBICI,   IF_REAL, "Internal BC voltage"),
 OP("vciei",    HICUM_QUEST_VCIEI,   IF_REAL, "Internal CE voltage"),
/* currents */
 OP("ic",     HICUM_QUEST_CC,    IF_REAL, "Collector current"),
 OP("iavl",   HICUM_QUEST_CAVL,  IF_REAL, "Avalanche current"),
 OP("ib",     HICUM_QUEST_CB,    IF_REAL, "Base current"),
 OP("ibei",   HICUM_QUEST_CBEI,    IF_REAL, "Intenral Base Emitter current"),
 OP("ibci",   HICUM_QUEST_CBCI,    IF_REAL, "Internal Base Collector current"),
 OP("ie",     HICUM_QUEST_CE,    IF_REAL, "Emitter current"),
 OP("is",     HICUM_QUEST_CS,    IF_REAL, "Substrate current"),
/* resistances */
 OP("rcx_t",  HICUM_QUEST_RCX_T, IF_REAL, "External (saturated) collector series resistance"),
 OP("re_t",   HICUM_QUEST_RE_T,  IF_REAL, "Emitter series resistance"),
 OP("rbi",    HICUM_QUEST_RBI,   IF_REAL, "Internal base resistance as calculated in the model"),
 OP("rb",     HICUM_QUEST_RB,    IF_REAL, "Total base resistance as calculated in the model"),

/* transconductances and capacitances */
 OP("betadc", HICUM_QUEST_BETADC, IF_REAL, "Common emitter forward current gain"),
 OP("gmi",    HICUM_QUEST_GMI,    IF_REAL, "Internal transconductance"),
 OP("gms",    HICUM_QUEST_GMS,    IF_REAL, "Transconductance of the parasitic substrate PNP"),
 OP("rpii",   HICUM_QUEST_RPII, IF_REAL, "Internal base-emitter (input) resistance"),
 OP("rpix",   HICUM_QUEST_RPIX, IF_REAL, "External base-emitter (input) resistance"),
 OP("rmui",   HICUM_QUEST_RMUI, IF_REAL, "Internal feedback resistance"),
 OP("rmux",   HICUM_QUEST_RMUX, IF_REAL, "External feedback resistance"),
 OP("roi",    HICUM_QUEST_ROI, IF_REAL, "Output resistance"),
 OP("cpii",   HICUM_QUEST_CPII, IF_REAL, "Total internal BE capacitance"),
 OP("cpix",   HICUM_QUEST_CPIX, IF_REAL, "Total external BE capacitance"),
 OP("cmui",   HICUM_QUEST_CMUI, IF_REAL, "Total internal BC capacitance"),
 OP("cmux",   HICUM_QUEST_CMUX, IF_REAL, "Total external BC capacitance"),
 OP("ccs",    HICUM_QUEST_CCS, IF_REAL, "CS junction capacitance"),
 OP("betaac", HICUM_QUEST_BETAAC, IF_REAL, "Small signal current gain"),
 OP("crbi",   HICUM_QUEST_CRBI, IF_REAL, "Shunt capacitance across RBI as calculated in the model"),
/* transit time */
 OP("tf",     HICUM_QUEST_TF,    IF_REAL, "Forward transit time"),
 OP("ft",     HICUM_QUEST_FT,    IF_REAL, "Transit frequency"),
 OP("ick",    HICUM_QUEST_ICK,   IF_REAL, "Transit frequency"),
/* power */
 OP("p",      HICUM_QUEST_POWER, IF_REAL, "Power dissipation")
};

IFparm HICUMmPTable[] = { /* model parameters */
//Circuit simulator specific parameters
  IOP("type",  HICUM_MOD_TYPE, IF_STRING, "For transistor type NPN(+1) or PNP (-1)"),
  IOPU("npn",  HICUM_MOD_NPN,  IF_FLAG, "NPN type device"),
  IOPU("pnp",  HICUM_MOD_PNP,  IF_FLAG, "PNP type device"),
  IOP("tnom",  HICUM_MOD_TNOM, IF_REAL, "Temperature at which parameters are specified"),
  IOPR("tref", HICUM_MOD_TNOM, IF_REAL, "Temperature at which parameters are specified"),
  IOP("version", HICUM_MOD_VERSION, IF_STRING, " parameter for model version"),

//Transfer current
  IOP("c10",   HICUM_MOD_C10  , IF_REAL, "GICCR constant"),
  IOP("qp0",   HICUM_MOD_QP0  , IF_REAL, "Zero-bias hole charge"),
  IOP("ich",   HICUM_MOD_ICH  , IF_REAL, "High-current correction for 2D and 3D effects"), //`0' signifies infinity
  IOP("hf0",   HICUM_MOD_HF0  , IF_REAL, "Weight factor for the low current minority charge"),
  IOP("hfe",   HICUM_MOD_HFE  , IF_REAL, "Emitter minority charge weighting factor in HBTs"),
  IOP("hfc",   HICUM_MOD_HFC  , IF_REAL, "Collector minority charge weighting factor in HBTs"),
  IOP("hjei",  HICUM_MOD_HJEI , IF_REAL, "B-E depletion charge weighting factor in HBTs"),
  IOP("ahjei", HICUM_MOD_AHJEI, IF_REAL, "Parameter describing the slope of hjEi(VBE)"),
  IOP("rhjei", HICUM_MOD_RHJEI, IF_REAL, "Smoothing parameter for hjEi(VBE) at high voltage"),
  IOP("hjci",  HICUM_MOD_HJCI , IF_REAL, "B-C depletion charge weighting factor in HBTs"),

//Base-Emitter diode currents
  IOP("ibeis", HICUM_MOD_IBEIS, IF_REAL, "Internal B-E saturation current"),
  IOP("mbei",  HICUM_MOD_MBEI , IF_REAL, "Internal B-E current ideality factor"),
  IOP("ireis", HICUM_MOD_IREIS, IF_REAL, "Internal B-E recombination saturation current"),
  IOP("mrei",  HICUM_MOD_MREI , IF_REAL, "Internal B-E recombination current ideality factor"),
  IOP("ibeps", HICUM_MOD_IBEPS, IF_REAL, "Peripheral B-E saturation current"),
  IOP("mbep",  HICUM_MOD_MBEP , IF_REAL, "Peripheral B-E current ideality factor"),
  IOP("ireps", HICUM_MOD_IREPS, IF_REAL, "Peripheral B-E recombination saturation current"),
  IOP("mrep",  HICUM_MOD_MREP , IF_REAL, "Peripheral B-E recombination current ideality factor"),
  IOP("mcf",   HICUM_MOD_MCF  , IF_REAL, "Non-ideality factor for III-V HBTs"),

//Transit time for excess recombination current at b-c barrier
  IOP("tbhrec", HICUM_MOD_TBHREC, IF_REAL, "Base current recombination time constant at B-C barrier for high forward injection"),

//Base-Collector diode currents
  IOP("ibcis", HICUM_MOD_IBCIS, IF_REAL, "Internal B-C saturation current"),
  IOP("mbci",  HICUM_MOD_MBCI , IF_REAL, "Internal B-C current ideality factor"),
  IOP("ibcxs", HICUM_MOD_IBCXS, IF_REAL, "External B-C saturation current"),
  IOP("mbcx",  HICUM_MOD_MBCX , IF_REAL, "External B-C current ideality factor"),

//Base-Emitter tunneling current
  IOP("ibets", HICUM_MOD_IBETS, IF_REAL, "B-E tunneling saturation current"),
  IOP("abet",  HICUM_MOD_ABET, IF_REAL, "Exponent factor for tunneling current"),
  IOP("tunode",HICUM_MOD_TUNODE, IF_INTEGER, "Specifies the base node connection for the tunneling current"), // =1 signifies perimeter node

//Base-Collector avalanche current
  IOP("favl",  HICUM_MOD_FAVL , IF_REAL, "Avalanche current factor"),
  IOP("qavl",  HICUM_MOD_QAVL , IF_REAL, "Exponent factor for avalanche current"),
  IOP("kavl",  HICUM_MOD_KAVL , IF_REAL, "Flag/factor for turning strong avalanche on"),
  IOP("alfav", HICUM_MOD_ALFAV, IF_REAL, "Relative TC for FAVL"),
  IOP("alqav", HICUM_MOD_ALQAV, IF_REAL, "Relative TC for QAVL"),
  IOP("alkav", HICUM_MOD_ALKAV, IF_REAL, "Relative TC for KAVL"),

//Series resistances
  IOP("rbi0",  HICUM_MOD_RBI0 , IF_REAL, "Zero bias internal base resistance"),
  IOP("rbx",   HICUM_MOD_RBX  , IF_REAL, "External base series resistance"),
  IOP("fgeo",  HICUM_MOD_FGEO , IF_REAL, "Factor for geometry dependence of emitter current crowding"),
  IOP("fdqr0", HICUM_MOD_FDQR0, IF_REAL, "Correction factor for modulation by B-E and B-C space charge layer"),
  IOP("fcrbi", HICUM_MOD_FCRBI, IF_REAL, "Ratio of HF shunt to total internal capacitance (lateral NQS effect)"),
  IOP("fqi",   HICUM_MOD_FQI  , IF_REAL, "Ration of internal to total minority charge"),
  IOP("re",    HICUM_MOD_RE   , IF_REAL, "Emitter series resistance"),
  IOP("rcx",   HICUM_MOD_RCX  , IF_REAL, "External collector series resistance"),

//Substrate transistor
  IOP("itss",  HICUM_MOD_ITSS, IF_REAL, "Substrate transistor transfer saturation current"),
  IOP("msf",   HICUM_MOD_MSF , IF_REAL, "Forward ideality factor of substrate transfer current"),
  IOP("iscs",  HICUM_MOD_ISCS, IF_REAL, "C-S diode saturation current"),
  IOP("msc",   HICUM_MOD_MSC , IF_REAL, "Ideality factor of C-S diode current"),
  IOP("tsf",   HICUM_MOD_TSF , IF_REAL, "Transit time for forward operation of substrate transistor"),

//Intra-device substrate coupling
  IOP("rsu",   HICUM_MOD_RSU, IF_REAL, "Substrate series resistance"),
  IOP("csu",   HICUM_MOD_CSU, IF_REAL, "Substrate shunt capacitance"),

//Depletion Capacitances
  IOP("cjei0",  HICUM_MOD_CJEI0 , IF_REAL, "Internal B-E zero-bias depletion capacitance"),
  IOP("vdei",   HICUM_MOD_VDEI  , IF_REAL, "Internal B-E built-in potential"),
  IOP("zei",    HICUM_MOD_ZEI   , IF_REAL, "Internal B-E grading coefficient"),
  IOP("ajei",   HICUM_MOD_AJEI  , IF_REAL, "Ratio of maximum to zero-bias value of internal B-E capacitance"),
  IOPR("aljei", HICUM_MOD_AJEI  , IF_REAL, "Ratio of maximum to zero-bias value of internal B-E capacitance"),
  IOP("cjep0",  HICUM_MOD_CJEP0 , IF_REAL, "Peripheral B-E zero-bias depletion capacitance"),
  IOP("vdep",   HICUM_MOD_VDEP  , IF_REAL, "Peripheral B-E built-in potential"),
  IOP("zep",    HICUM_MOD_ZEP   , IF_REAL, "Peripheral B-E grading coefficient"),
  IOP("ajep",   HICUM_MOD_AJEP  , IF_REAL, "Ratio of maximum to zero-bias value of peripheral B-E capacitance"),
  IOPR("aljep", HICUM_MOD_AJEP  , IF_REAL, "Ratio of maximum to zero-bias value of peripheral B-E capacitance"),
  IOP("cjci0",  HICUM_MOD_CJCI0 , IF_REAL, "Internal B-C zero-bias depletion capacitance"),
  IOP("vdci",   HICUM_MOD_VDCI  , IF_REAL, "Internal B-C built-in potential"),
  IOP("zci",    HICUM_MOD_ZCI   , IF_REAL, "Internal B-C grading coefficient"),
  IOP("vptci",  HICUM_MOD_VPTCI , IF_REAL, "Internal B-C punch-through voltage"),
  IOP("cjcx0",  HICUM_MOD_CJCX0 , IF_REAL, "External B-C zero-bias depletion capacitance"),
  IOP("vdcx",   HICUM_MOD_VDCX  , IF_REAL, "External B-C built-in potential"),
  IOP("zcx",    HICUM_MOD_ZCX   , IF_REAL, "External B-C grading coefficient"),
  IOP("vptcx",  HICUM_MOD_VPTCX , IF_REAL, "External B-C punch-through voltage"),
  IOP("fbcpar", HICUM_MOD_FBCPAR, IF_REAL, "Partitioning factor of parasitic B-C cap"),
  IOPR("fbc",   HICUM_MOD_FBCPAR, IF_REAL, "Partitioning factor of parasitic B-C cap"),
  IOP("fbepar", HICUM_MOD_FBEPAR, IF_REAL, "Partitioning factor of parasitic B-E cap"),
  IOPR("fbe",   HICUM_MOD_FBEPAR, IF_REAL, "Partitioning factor of parasitic B-E cap"),
  IOP("cjs0",   HICUM_MOD_CJS0  , IF_REAL, "C-S zero-bias depletion capacitance"),
  IOP("vds",    HICUM_MOD_VDS   , IF_REAL, "C-S built-in potential"),
  IOP("zs",     HICUM_MOD_ZS    , IF_REAL, "C-S grading coefficient"),
  IOP("vpts",   HICUM_MOD_VPTS  , IF_REAL, "C-S punch-through voltage"),
  IOP("cscp0",  HICUM_MOD_CSCP0 , IF_REAL, "Perimeter S-C zero-bias depletion capacitance"),
  IOP("vdsp",   HICUM_MOD_VDSP  , IF_REAL, "Perimeter S-C built-in potential"),
  IOP("zsp",    HICUM_MOD_ZSP   , IF_REAL, "Perimeter S-C grading coefficient"),
  IOP("vptsp",  HICUM_MOD_VPTSP , IF_REAL, "Perimeter S-C punch-through voltage"),

//Diffusion Capacitances
  IOP("t0",    HICUM_MOD_T0   , IF_REAL, "Low current forward transit time at VBC=0V"),
  IOP("dt0h",  HICUM_MOD_DT0H , IF_REAL, "Time constant for base and B-C space charge layer width modulation"),
  IOP("tbvl",  HICUM_MOD_TBVL , IF_REAL, "Time constant for modeling carrier jam at low VCE"),
  IOP("tef0",  HICUM_MOD_TEF0 , IF_REAL, "Neutral emitter storage time"),
  IOP("gtfe",  HICUM_MOD_GTFE , IF_REAL, "Exponent factor for current dependence of neutral emitter storage time"),
  IOP("thcs",  HICUM_MOD_THCS , IF_REAL, "Saturation time constant at high current densities"),
  IOP("ahc",   HICUM_MOD_AHC  , IF_REAL, "Smoothing factor for current dependence of base and collector transit time"),
  IOPR("alhc", HICUM_MOD_AHC  , IF_REAL, "Smoothing factor for current dependence of base and collector transit time"),
  IOP("fthc",  HICUM_MOD_FTHC , IF_REAL, "Partitioning factor for base and collector portion"),
  IOP("rci0",  HICUM_MOD_RCI0 , IF_REAL, "Internal collector resistance at low electric field"),
  IOP("vlim",  HICUM_MOD_VLIM , IF_REAL, "Voltage separating ohmic and saturation velocity regime"),
  IOP("vces",  HICUM_MOD_VCES , IF_REAL, "Internal C-E saturation voltage"),
  IOP("vpt",   HICUM_MOD_VPT  , IF_REAL, "Collector punch-through voltage"), // `0' signifies infinity
  IOP("aick",  HICUM_MOD_AICK , IF_REAL, "Smoothing term for ICK"),
  IOP("delck", HICUM_MOD_DELCK, IF_REAL, "Fitting factor for critical current"),
  IOP("tr",    HICUM_MOD_TR   , IF_REAL, "Storage time for inverse operation"),
  IOP("vcbar", HICUM_MOD_VCBAR, IF_REAL, "Barrier voltage"),
  IOP("icbar", HICUM_MOD_ICBAR, IF_REAL, "Normalization parameter"),
  IOP("acbar", HICUM_MOD_ACBAR, IF_REAL, "Smoothing parameter for barrier voltage"),

//Isolation Capacitances
  IOP("cbepar", HICUM_MOD_CBEPAR, IF_REAL, "Total parasitic B-E capacitance"),
  IOPR("ceox",  HICUM_MOD_CBEPAR, IF_REAL, "Total parasitic B-E capacitance"),
  IOP("cbcpar", HICUM_MOD_CBCPAR, IF_REAL, "Total parasitic B-C capacitance"),
  IOPR("ccox",  HICUM_MOD_CBCPAR, IF_REAL, "Total parasitic B-C capacitance"),

//Non-quasi-static Effect
  IOP("alqf",  HICUM_MOD_ALQF,  IF_REAL, "Factor for additional delay time of minority charge"),
  IOP("alit",  HICUM_MOD_ALIT,  IF_REAL, "Factor for additional delay time of transfer current"),
  IOP("flnqs", HICUM_MOD_FLNQS, IF_INTEGER, "Flag for turning on and off of vertical NQS effect"),

//Noise
  IOP("kf",     HICUM_MOD_KF    , IF_REAL, "Flicker noise coefficient"),
  IOP("af",     HICUM_MOD_AF    , IF_REAL, "Flicker noise exponent factor"),
  IOP("cfbe",   HICUM_MOD_CFBE  , IF_INTEGER, "Flag for determining where to tag the flicker noise source"),
  IOP("flcono", HICUM_MOD_FLCONO, IF_INTEGER, "Flag for turning on and off of correlated noise implementation"),
  IOP("kfre",   HICUM_MOD_KFRE  , IF_REAL, "Emitter resistance flicker noise coefficient"),
  IOP("afre",   HICUM_MOD_AFRE  , IF_REAL, "Emitter resistance flicker noise exponent factor"),

//Lateral Geometry Scaling (at high current densities)
  IOP("latb", HICUM_MOD_LATB, IF_REAL, "Scaling factor for collector minority charge in direction of emitter width"),
  IOP("latl", HICUM_MOD_LATL, IF_REAL, "Scaling factor for collector minority charge in direction of emitter length"),

//Temperature dependence
  IOP("vgb",      HICUM_MOD_VGB     , IF_REAL, "Bandgap voltage extrapolated to 0 K"),
  IOP("alt0",     HICUM_MOD_ALT0    , IF_REAL, "First order relative TC of parameter T0"),
  IOP("kt0",      HICUM_MOD_KT0     , IF_REAL, "Second order relative TC of parameter T0"),
  IOP("zetaci",   HICUM_MOD_ZETACI  , IF_REAL, "Temperature exponent for RCI0"),
  IOP("alvs",     HICUM_MOD_ALVS    , IF_REAL, "Relative TC of saturation drift velocity"),
  IOP("alces",    HICUM_MOD_ALCES   , IF_REAL, "Relative TC of VCES"),
  IOP("zetarbi",  HICUM_MOD_ZETARBI , IF_REAL, "Temperature exponent of internal base resistance"),
  IOP("zetarbx",  HICUM_MOD_ZETARBX , IF_REAL, "Temperature exponent of external base resistance"),
  IOP("zetarcx",  HICUM_MOD_ZETARCX , IF_REAL, "Temperature exponent of external collector resistance"),
  IOP("zetare",   HICUM_MOD_ZETARE  , IF_REAL, "Temperature exponent of emitter resistance"),
  IOP("zetacx",   HICUM_MOD_ZETACX  , IF_REAL, "Temperature exponent of mobility in substrate transistor transit time"),
  IOP("vge",      HICUM_MOD_VGE     , IF_REAL, "Effective emitter bandgap voltage"),
  IOP("vgc",      HICUM_MOD_VGC     , IF_REAL, "Effective collector bandgap voltage"),
  IOP("vgs",      HICUM_MOD_VGS     , IF_REAL, "Effective substrate bandgap voltage"),
  IOP("f1vg",     HICUM_MOD_F1VG    , IF_REAL, "Coefficient K1 in T-dependent band-gap equation"),
  IOP("f2vg",     HICUM_MOD_F2VG    , IF_REAL, "Coefficient K2 in T-dependent band-gap equation"),
  IOP("zetact",   HICUM_MOD_ZETACT  , IF_REAL, "Exponent coefficient in transfer current temperature dependence"),
  IOP("zetabet",  HICUM_MOD_ZETABET , IF_REAL, "Exponent coefficient in B-E junction current temperature dependence"),
  IOP("alb",      HICUM_MOD_ALB     , IF_REAL, "Relative TC of forward current gain for V2.1 model"),
  IOP("dvgbe",    HICUM_MOD_DVGBE   , IF_REAL, "Bandgap difference between B and B-E junction used for hjEi0 and hf0"),
  IOP("zetahjei", HICUM_MOD_ZETAHJEI, IF_REAL, "Temperature coefficient for ahjEi"),
  IOP("zetavgbe", HICUM_MOD_ZETAVGBE, IF_REAL, "Temperature coefficient for hjEi0"),

//Self-Heating
  IOP("flsh",    HICUM_MOD_FLSH   , IF_INTEGER, "Flag for turning on and off self-heating effect"),
  IOP("rth",     HICUM_MOD_RTH    , IF_REAL, "Thermal resistance"),
  IOP("zetarth", HICUM_MOD_ZETARTH, IF_REAL, "Temperature coefficient for Rth"),
  IOP("alrth",   HICUM_MOD_ALRTH  , IF_REAL, "First order relative TC of parameter Rth"),
  IOP("cth",     HICUM_MOD_CTH    , IF_REAL, "Thermal capacitance"),

//Compatibility with V2.1
  IOP("flcomp", HICUM_MOD_FLCOMP, IF_REAL, "Flag for compatibility with v2.1 model (0=v2.1)"),

  IOP("vbe_max", HICUM_MOD_VBE_MAX, IF_REAL, "maximum voltage B-E junction"),
  IOP("vbc_max", HICUM_MOD_VBC_MAX, IF_REAL, "maximum voltage B-C junction"),
  IOP("vce_max", HICUM_MOD_VCE_MAX, IF_REAL, "maximum voltage C-E branch")

};

char *HICUMnames[] = {
    "collector",
    "base",
    "emitter",
    "substrate",
    "temp"
};


int HICUMnSize = NUMELEMS(HICUMnames);
int HICUMpTSize = NUMELEMS(HICUMpTable);
int HICUMmPTSize = NUMELEMS(HICUMmPTable);
int HICUMiSize = sizeof(HICUMinstance);
int HICUMmSize = sizeof(HICUMmodel);
