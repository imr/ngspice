/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "soi3defs.h"
#include "ngspice/suffix.h"


char *SOI3names[] = {
    "Drain",
    "Front Gate",
    "Source",
    "Back Gate",
    "Bulk",
    "Thermal"
};


IFparm SOI3pTable[] = { /* parameters */
 IOP("l",            SOI3_L,          IF_REAL,    "Length"),
 IOP("w",            SOI3_W,          IF_REAL,    "Width"),
 IOP("m",            SOI3_M,          IF_REAL,    "Parallel Multiplier"),
 IOP("as",           SOI3_AS,         IF_REAL,	  "Source area"),
 IOP("ad",           SOI3_AD,         IF_REAL,	  "Drain area"),
 IOP("ab",           SOI3_AB,         IF_REAL,	  "Body area"),
 IOP("nrd",          SOI3_NRD,        IF_REAL,    "Drain squares"),
 IOP("nrs",          SOI3_NRS,        IF_REAL,    "Source squares"),
 IP("off",           SOI3_OFF,        IF_FLAG,    "Device initially off"),
 IOP("icvds",        SOI3_IC_VDS,     IF_REAL,    "Initial D-S voltage"),
 IOP("icvgfs",       SOI3_IC_VGFS,    IF_REAL,    "Initial GF-S voltage"),
 IOP("icvgbs",       SOI3_IC_VGBS,    IF_REAL,    "Initial GB-S voltage"),
 IOP("icvbs",        SOI3_IC_VBS,     IF_REAL,    "Initial B-S voltage"),
 IOP("temp",         SOI3_TEMP,       IF_REAL,    "Instance temperature"),
 IOP("rt",	         SOI3_RT,	        IF_REAL,    "Instance Lumped Thermal Resistance"),
 IOP("ct",	         SOI3_CT,	        IF_REAL,    "Instance Lumped Thermal Capacitance"),
 IOP("rt1",	         SOI3_RT1,	     IF_REAL,    "Second Thermal Resistance"),
 IOP("ct1",	         SOI3_CT1,	     IF_REAL,    "Second Thermal Capacitance"),
 IOP("rt2",	         SOI3_RT2,	     IF_REAL,    "Third Thermal Resistance"),
 IOP("ct2",	         SOI3_CT2,	     IF_REAL,    "Third Thermal Capacitance"),
 IOP("rt3",	         SOI3_RT3,	     IF_REAL,    "Fourth Thermal Resistance"),
 IOP("ct3",	         SOI3_CT3,	     IF_REAL,    "Fourth Thermal Capacitance"),
 IOP("rt4",	         SOI3_RT4,	     IF_REAL,    "Fifth Thermal Resistance"),
 IOP("ct4",	         SOI3_CT4,	     IF_REAL,    "Fifth Thermal Capacitance"),

 IP( "ic", SOI3_IC, IF_REALVEC,"Vector of D-S, GF-S, GB-S, B-S voltages"),
/* IP( "sens_l", SOI3_L_SENS, IF_FLAG, "flag to request sensitivity WRT length"),
 IP( "sens_w", SOI3_W_SENS, IF_FLAG, "flag to request sensitivity WRT width"),*/
 OP( "dnode",        SOI3_DNODE,      IF_INTEGER, "Number of the drain node "),
 OP( "gfnode",       SOI3_GFNODE,     IF_INTEGER, "Number of frt. gate node "),
 OP( "snode",        SOI3_SNODE,      IF_INTEGER, "Number of the source node "),
 OP( "gbnode",       SOI3_GBNODE,     IF_INTEGER, "Number of back gate node "),
 OP( "bnode",        SOI3_BNODE,      IF_INTEGER, "Number of the body node "),
 OP( "dnodeprime", SOI3_DNODEPRIME, IF_INTEGER, "Number of int. drain node"),
 OP( "snodeprime", SOI3_SNODEPRIME, IF_INTEGER, "Number of int. source node "),
 OP( "tnode",        SOI3_TNODE,    IF_INTEGER, "Number of thermal node "),
 OP( "branch",       SOI3_BRANCH,   IF_INTEGER, "Number of thermal branch "),
 OP( "sourceconductance", SOI3_SOURCECONDUCT, IF_REAL, "Conductance of source"),
 OP( "drainconductance",  SOI3_DRAINCONDUCT,  IF_REAL, "Conductance of drain"),
 OP( "von",          SOI3_VON,        IF_REAL,    "Effective Threshold Voltage (von) "),
 OP( "vfbf",         SOI3_VFBF,       IF_REAL,    "Temperature adjusted flat band voltage"),
 OP( "vdsat",        SOI3_VDSAT,      IF_REAL,    "Saturation drain voltage"),
 OP( "sourcevcrit",  SOI3_SOURCEVCRIT,IF_REAL,    "Critical source voltage"),
 OP( "drainvcrit",   SOI3_DRAINVCRIT, IF_REAL,    "Critical drain voltage"),
 OP( "id",           SOI3_ID,         IF_REAL,    "Drain current"),
 OP( "ibs",      SOI3_IBS,    IF_REAL,    "B-S junction current"),
 OP( "ibd",      SOI3_IBD,    IF_REAL,    "B-D junction current"),
 OP( "gmbs",     SOI3_GMBS,   IF_REAL,    "Bulk-Source transconductance"),
 OP( "gmf",          SOI3_GMF,        IF_REAL,    "Front transconductance"),
 OP( "gmb",          SOI3_GMB,        IF_REAL,    "Back transconductance"),
 OP( "gds",          SOI3_GDS,        IF_REAL,    "Drain-Source conductance"),
 OP( "gbd",          SOI3_GBD,        IF_REAL,    "Bulk-Drain conductance"),
 OP( "gbs",          SOI3_GBS,        IF_REAL,    "Bulk-Source conductance"),
 OP( "capbd",        SOI3_CAPBD,      IF_REAL,    "Bulk-Drain capacitance"),
 OP( "capbs",        SOI3_CAPBS,      IF_REAL,    "Bulk-Source capacitance"),
 OP( "cbd0", SOI3_CAPZEROBIASBD, IF_REAL, "Zero-Bias B-D junction capacitance"),
 OP( "cbs0", SOI3_CAPZEROBIASBS, IF_REAL, "Zero-Bias B-S junction capacitance"),
 OP( "vbd",          SOI3_VBD,        IF_REAL,    "Bulk-Drain voltage"),
 OP( "vbs",          SOI3_VBS,        IF_REAL,    "Bulk-Source voltage"),
 OP( "vgfs",         SOI3_VGFS,       IF_REAL, "Front gate-Source voltage"),
 OP( "vgbs",         SOI3_VGBS,       IF_REAL, "Back gate-Source voltage"),
 OP( "vds",          SOI3_VDS,        IF_REAL,    "Drain-Source voltage"),
 OP( "qgf",          SOI3_QGF,        IF_REAL,    "Front Gate charge storage"),
 OP( "iqgf",SOI3_IQGF,IF_REAL,"Current due to front gate charge storage"),
/*
 OP( "qgb",          SOI3_QGB,        IF_REAL,    "Back Gate charge storage"),
 OP( "iqgb",SOI3_IQGB,IF_REAL,"Current due to back gate charge storage"),
*/
 OP( "qd",           SOI3_QD,         IF_REAL,    "Drain charge storage"),
 OP( "iqd",SOI3_IQD,IF_REAL,"Current due to drain charge storage"),
 OP( "qs",           SOI3_QS,         IF_REAL,    "Source charge storage"),
 OP( "iqs",SOI3_IQS,IF_REAL,"Current due to source charge storage"), 
 OP( "qbd",          SOI3_QBD,        IF_REAL,    "Bulk-Drain charge storage"),
 OP( "iqbd",SOI3_IQBD,IF_REAL,"Current due to bulk-drain charge storage"),
 OP( "qbs",          SOI3_QBS,        IF_REAL,    "Bulk-Source charge storage"),
 OP( "iqbs",SOI3_IQBS,IF_REAL,"Currnet due to bulk-source charge storage"),
/* extra stuff for newer model -msll Jan96 */
 OP( "vfbb",         SOI3_VFBB,       IF_REAL,    "Temperature adjusted back flat band voltage")

/*,
 OP( "sens_l_dc",    SOI3_L_SENS_DC,  IF_REAL,    "dc sensitivity wrt length"),
 OP( "sens_l_real", SOI3_L_SENS_REAL,IF_REAL,
        "real part of ac sensitivity wrt length"),
 OP( "sens_l_imag",  SOI3_L_SENS_IMAG,IF_REAL,
        "imag part of ac sensitivity wrt length"),
 OP( "sens_l_mag",   SOI3_L_SENS_MAG, IF_REAL,
        "sensitivity wrt l of ac magnitude"),
 OP( "sens_l_ph",    SOI3_L_SENS_PH,  IF_REAL,
        "sensitivity wrt l of ac phase"),
 OP( "sens_l_cplx",  SOI3_L_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt length"),
 OP( "sens_w_dc",    SOI3_W_SENS_DC,  IF_REAL,    "dc sensitivity wrt width"),
 OP( "sens_w_real",  SOI3_W_SENS_REAL,IF_REAL,
        "real part of ac sensitivity wrt width"),
 OP( "sens_w_imag",  SOI3_W_SENS_IMAG,IF_REAL,
        "imag part of ac sensitivity wrt width"),
 OP( "sens_w_mag",   SOI3_W_SENS_MAG, IF_REAL,
        "sensitivity wrt w of ac magnitude"),
 OP( "sens_w_ph",    SOI3_W_SENS_PH,  IF_REAL,
        "sensitivity wrt w of ac phase"),
 OP( "sens_w_cplx",  SOI3_W_SENS_CPLX,IF_COMPLEX, "ac sensitivity wrt width")*/
};

IFparm SOI3mPTable[] = { /* model parameters */
 IOP("vto",   SOI3_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOPR("vt0",  SOI3_MOD_VTO,   IF_REAL   ,"Threshold voltage"),
 IOP("vfbf",  SOI3_MOD_VFBF,  IF_REAL   ,"Flat band voltage"),
 IOP("kp",    SOI3_MOD_KP,    IF_REAL   ,"Transconductance parameter"),
 IOP("gamma", SOI3_MOD_GAMMA, IF_REAL   ,"Body Factor"),
 IOP("phi",   SOI3_MOD_PHI,   IF_REAL   ,"Surface potential"),
 IOP("lambda",SOI3_MOD_LAMBDA,IF_REAL   ,"Channel length modulation"),
 IOP("theta", SOI3_MOD_THETA, IF_REAL   ,"Vertical field mobility degradation"),
 IOP("rd",    SOI3_MOD_RD,    IF_REAL   ,"Drain ohmic resistance"),
 IOP("rs",    SOI3_MOD_RS,    IF_REAL   ,"Source ohmic resistance"),
 IOP("cbd",   SOI3_MOD_CBD,   IF_REAL   ,"B-D junction capacitance"),
 IOP("cbs",   SOI3_MOD_CBS,   IF_REAL   ,"B-S junction capacitance"),
 IOP("is",    SOI3_MOD_IS,    IF_REAL   ,"Bulk junction sat. current"),
 IOP("is1",   SOI3_MOD_IS1,   IF_REAL   ,"2nd Bulk junction sat. current"),
 IOP("pb",    SOI3_MOD_PB,    IF_REAL   ,"Bulk junction potential"),
 IOP("cgfso",  SOI3_MOD_CGFSO,  IF_REAL   ,"Front Gate-source overlap cap."),
 IOP("cgfdo",  SOI3_MOD_CGFDO,  IF_REAL   ,"Front Gate-drain overlap cap."),
 IOP("cgfbo",  SOI3_MOD_CGFBO,  IF_REAL   ,"Front Gate-bulk overlap cap."),
 IOP("cgbso",  SOI3_MOD_CGBSO,  IF_REAL   ,"Back Gate-source overlap cap."),
 IOP("cgbdo",  SOI3_MOD_CGBDO,  IF_REAL   ,"Back Gate-drain overlap cap."),
 IOP("cgbbo",  SOI3_MOD_CGBBO,  IF_REAL   ,"Back Gate-bulk overlap cap."),
 IOP("rsh",   SOI3_MOD_RSH,   IF_REAL   ,"Sheet resistance"),
 IOP("cj",    SOI3_MOD_CJSW,  IF_REAL   ,"Side junction cap per area"),
 IOP("mj",    SOI3_MOD_MJSW,  IF_REAL   ,"Side grading coefficient"),
 IOP("js",    SOI3_MOD_JS,    IF_REAL   ,"Bulk jct. sat. current density"),
 IOP("js1",   SOI3_MOD_JS1,   IF_REAL   ,"2nd Bulk jct. sat. current density"),
 IOP("tof",   SOI3_MOD_TOF,   IF_REAL   ,"Front Oxide thickness"),
 IOP("tob",   SOI3_MOD_TOB,   IF_REAL   ,"Back Oxide thickness"),
 IOP("tb",    SOI3_MOD_TB,    IF_REAL   ,"Bulk film thickness"),
 IOP("ld",    SOI3_MOD_LD,    IF_REAL   ,"Lateral diffusion"),
 IOP("u0",    SOI3_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOPR("uo",   SOI3_MOD_U0,    IF_REAL   ,"Surface mobility"),
 IOP("fc",    SOI3_MOD_FC,    IF_REAL   ,"Forward bias jct. fit parm."),
 IP("nsoi",   SOI3_MOD_NSOI3,  IF_FLAG   ,"N type SOI3fet model"),
 IP("psoi",   SOI3_MOD_PSOI3,  IF_FLAG   ,"P type SOI3fet model"),
 IOP("kox",   SOI3_MOD_KOX,   IF_REAL   ,"Oxide thermal conductivity"),
 IOP("shsi",  SOI3_MOD_SHSI,  IF_REAL   ,"Specific heat of silicon"),
 IOP("dsi",   SOI3_MOD_DSI,   IF_REAL   ,"Density of silicon"),
 IOP("nsub",  SOI3_MOD_NSUB,  IF_REAL   ,"Substrate doping"),
 IOP("tpg",   SOI3_MOD_TPG,   IF_INTEGER,"Gate type"),
 IOP("nqff",  SOI3_MOD_NQFF,  IF_REAL   ,"Front fixed oxide charge density"),
 IOP("nqfb",  SOI3_MOD_NQFB,  IF_REAL   ,"Back fixed oxide charge density"),
 IOP("nssf",  SOI3_MOD_NSSF,  IF_REAL   ,"Front surface state density"),
 IOP("nssb",  SOI3_MOD_NSSB,  IF_REAL   ,"Back surface state density"),
 IOP("tnom",  SOI3_MOD_TNOM,  IF_REAL   ,"Parameter measurement temp"),
 IP("kf",     SOI3_MOD_KF,    IF_REAL   ,"Flicker noise coefficient"),
 IP("af",     SOI3_MOD_AF,    IF_REAL   ,"Flicker noise exponent"),
/* extra stuff for newer model - msll Jan96 */
 IOP("sigma", SOI3_MOD_SIGMA, IF_REAL   ,"DIBL coefficient"),
 IOP("chifb", SOI3_MOD_CHIFB, IF_REAL   ,"Temperature coeff of flatband voltage"),
 IOP("chiphi",SOI3_MOD_CHIPHI, IF_REAL   ,"Temperature coeff of PHI"),
 IOP("deltaw",SOI3_MOD_DELTAW,IF_REAL   ,"Narrow width factor"),
 IOP("deltal",SOI3_MOD_DELTAL,IF_REAL   ,"Short channel factor"),
 IOP("vsat",  SOI3_MOD_VSAT,  IF_REAL   ,"Saturation velocity"),
 IOP("k",     SOI3_MOD_K,     IF_REAL   ,"Thermal exponent"),
 IOP("lx",    SOI3_MOD_LX,    IF_REAL   ,"Channel length modulation (alternative)"),
 IOP("vp",    SOI3_MOD_VP,    IF_REAL   ,"Channel length modulation (alt. empirical)"),
 IOP("eta",   SOI3_MOD_ETA,   IF_REAL   ,"Impact ionization field adjustment factor"),
 IOP("alpha0",SOI3_MOD_ALPHA0,IF_REAL   ,"First impact ionisation coeff (alpha0)"),
 IOP("beta0", SOI3_MOD_BETA0, IF_REAL   ,"Second impact ionisation coeff (beta0)"),
 IOP("lm",    SOI3_MOD_LM,    IF_REAL   ,"Impact ion. drain region length"),
 IOP("lm1",   SOI3_MOD_LM1,   IF_REAL   ,"Impact ion. drain region length coeff"),
 IOP("lm2",   SOI3_MOD_LM2,   IF_REAL   ,"Impact ion. drain region length coeff"),
 IOP("etad",  SOI3_MOD_ETAD,  IF_REAL   ,"Diode ideality factor"),
 IOP("etad1", SOI3_MOD_ETAD1, IF_REAL   ,"2nd Diode ideality factor"),
 IOP("chibeta",SOI3_MOD_CHIBETA,IF_REAL ,"Impact ionisation temperature coefficient"),
 IOP("vfbb",  SOI3_MOD_VFBB,  IF_REAL   ,"Back Flat band voltage"),
 IOP("gammab",SOI3_MOD_GAMMAB,IF_REAL   ,"Back Body Factor"),
 IOP("chid",  SOI3_MOD_CHID,  IF_REAL   ,"Junction temperature factor"),
 IOP("chid1", SOI3_MOD_CHID1, IF_REAL   ,"2nd Junction temperature factor"),
 IOP("dvt",   SOI3_MOD_DVT,   IF_INTEGER,"Switch for temperature dependence of vt in diodes"),
 IOP("nlev",  SOI3_MOD_NLEV,  IF_INTEGER,"Level switch for flicker noise model"),
 IOP("betabjt",SOI3_MOD_BETABJT,IF_REAL ,"Beta for BJT"),
 IOP("tauf",  SOI3_MOD_TAUFBJT,IF_REAL  ,"Forward tau for BJT"),
 IOP("taur",  SOI3_MOD_TAURBJT,IF_REAL  ,"Reverse tau for BJT"),
 IOP("betaexp",SOI3_MOD_BETAEXP,IF_REAL ,"Exponent for Beta of BJT"),
 IOP("tauexp", SOI3_MOD_TAUEXP,IF_REAL,  "Exponent for Transit time of BJT"),
 IOP("rsw",   SOI3_MOD_RSW,   IF_REAL   ,"Source resistance width scaling factor"),
 IOP("rdw",   SOI3_MOD_RDW,   IF_REAL   ,"Drain resistance width scaling factor"),
 IOP("fmin",  SOI3_MOD_FMIN,  IF_REAL   ,"Minimum feature size of technology"),
 IOP("vtex",  SOI3_MOD_VTEX,  IF_REAL   ,"Extracted threshold voltage"),
 IOP("vdex",  SOI3_MOD_VDEX,  IF_REAL   ,"Drain bias at which vtex extracted"),
 IOP("delta0",SOI3_MOD_DELTA0,IF_REAL   ,"Surface potential factor for vtex conversion"),
 IOP("csf",   SOI3_MOD_CSF   ,IF_REAL   ,"Saturation region charge sharing factor"),
 IOP("nplus", SOI3_MOD_NPLUS  ,IF_REAL  ,"Doping concentration of N+ or P+ regions"),
 IOP("rta",   SOI3_MOD_RTA    ,IF_REAL  ,"Thermal resistance area scaling factor"),
 IOP("cta",   SOI3_MOD_CTA    ,IF_REAL  ,"Thermal capacitance area scaling factor"),
 IOP("mexp",  SOI3_MOD_MEXP   ,IF_REAL  ,"Exponent for CLM smoothing")
};


int     SOI3nSize = NUMELEMS(SOI3names);
int     SOI3pTSize = NUMELEMS(SOI3pTable);
int     SOI3mPTSize = NUMELEMS(SOI3mPTable);
int     SOI3iSize = sizeof(SOI3instance);
int     SOI3mSize = sizeof(SOI3model);

