/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * This file defines the BJT2 data structures that are
 * available to the next level(s) up the calling hierarchy
 */

/*
 * You may define the preprocessor symbolo
 * BJT2_COMPAT to enable compatibility with
 * archaic spice2 bjt model
 */ 

#include "ngspice.h"
#include "devdefs.h"
#include "bjt2defs.h"
#include "suffix.h"

IFparm BJT2pTable[] = { /* parameters */
 IOPU("off",     BJT2_OFF,            IF_FLAG,    "Device initially off"),
 IOPAU("icvbe",  BJT2_IC_VBE,         IF_REAL, "Initial B-E voltage"),
 IOPAU("icvce",  BJT2_IC_VCE,         IF_REAL, "Initial C-E voltage"),
 IOPU("area",    BJT2_AREA,           IF_REAL,    "(Emitter) Area factor"),
 IOPU("areab",   BJT2_AREAB,          IF_REAL,    "Base area factor"),
 IOPU("areac",   BJT2_AREAC,          IF_REAL,    "Collector area factor"), 
 IOPU("m",       BJT2_M,              IF_REAL,   "Parallel Multiplier"),
 IP("ic",       BJT2_IC,             IF_REALVEC, "Initial condition vector"),
 IP("sens_area",BJT2_AREA_SENS,IF_FLAG, "flag to request sensitivity WRT area"),
 OPU("colnode",  BJT2_QUEST_COLNODE,  IF_INTEGER, "Number of collector node"),
 OPU("basenode", BJT2_QUEST_BASENODE, IF_INTEGER, "Number of base node"),
 OPU("emitnode", BJT2_QUEST_EMITNODE, IF_INTEGER, "Number of emitter node"),
 OPU("substnode",BJT2_QUEST_SUBSTNODE,IF_INTEGER, "Number of substrate node"),
 OPU("colprimenode",BJT2_QUEST_COLPRIMENODE,IF_INTEGER,
						"Internal collector node"),
 OPU("baseprimenode",BJT2_QUEST_BASEPRIMENODE,IF_INTEGER,"Internal base node"),
 OPU("emitprimenode",BJT2_QUEST_EMITPRIMENODE,IF_INTEGER,
						"Internal emitter node"),
 OP("ic",    BJT2_QUEST_CC,   IF_REAL, "Current at collector node"),
 OP("ib",    BJT2_QUEST_CB,   IF_REAL, "Current at base node"),
 OP("ie",   BJT2_QUEST_CE,   IF_REAL, "Emitter current"),
 OPU("is",   BJT2_QUEST_CS,   IF_REAL, "Substrate current"),
 OP("vbe",   BJT2_QUEST_VBE,  IF_REAL, "B-E voltage"),
 OP("vbc",   BJT2_QUEST_VBC,  IF_REAL, "B-C voltage"),
 OP("gm",    BJT2_QUEST_GM,   IF_REAL, "Small signal transconductance"),
 OP("gpi",   BJT2_QUEST_GPI,  IF_REAL, "Small signal input conductance - pi"),
 OP("gmu",   BJT2_QUEST_GMU,  IF_REAL, "Small signal conductance - mu"),
 OP("gx",   BJT2_QUEST_GX,   IF_REAL, "Conductance from base to internal base"),
 OP("go",    BJT2_QUEST_GO,   IF_REAL, "Small signal output conductance"),
 OPU("geqcb",BJT2_QUEST_GEQCB,IF_REAL, "d(Ibe)/d(Vbc)"),
 OPU("gcsub", BJT2_QUEST_GCSUB, IF_REAL, "Internal Subs. cap. equiv. cond."),
 OPU("gdsub", BJT2_QUEST_GDSUB, IF_REAL, "Internal Subs. Diode equiv. cond."),
 OPU("geqbx",BJT2_QUEST_GEQBX,IF_REAL, "Internal C-B-base cap. equiv. cond."),

 OP("cpi",BJT2_QUEST_CPI, IF_REAL, "Internal base to emitter capactance"),
 OP("cmu",BJT2_QUEST_CMU, IF_REAL, "Internal base to collector capactiance"),
 OP("cbx",BJT2_QUEST_CBX, IF_REAL, "Base to collector capacitance"),
 OP("csub",BJT2_QUEST_CSUB, IF_REAL, "Substrate capacitance"),
 OPU("cqbe",BJT2_QUEST_CQBE, IF_REAL, "Cap. due to charge storage in B-E jct."),
 OPU("cqbc",BJT2_QUEST_CQBC, IF_REAL, "Cap. due to charge storage in B-C jct."),
 OPU("cqsub", BJT2_QUEST_CQSUB, IF_REAL, "Cap. due to charge storage in Subs. jct."),
 OPU("cqbx", BJT2_QUEST_CQBX, IF_REAL, "Cap. due to charge storage in B-X jct."),
 OPU("cexbc",BJT2_QUEST_CEXBC,IF_REAL, "Total Capacitance in B-X junction"),

 OPU("qbe",  BJT2_QUEST_QBE,  IF_REAL, "Charge storage B-E junction"),
 OPU("qbc",   BJT2_QUEST_QBC,  IF_REAL, "Charge storage B-C junction"),
 OPU("qsub",  BJT2_QUEST_QSUB,  IF_REAL, "Charge storage Subs. junction"),
 OPU("qbx",  BJT2_QUEST_QBX,  IF_REAL, "Charge storage B-X junction"),
 OPU("p",    BJT2_QUEST_POWER,IF_REAL, "Power dissipation"),
 OPU("sens_dc", BJT2_QUEST_SENS_DC, IF_REAL,    "dc sensitivity "),
 OPU("sens_real", BJT2_QUEST_SENS_REAL, IF_REAL,"real part of ac sensitivity"),
 OPU("sens_imag",BJT2_QUEST_SENS_IMAG,IF_REAL,
					"dc sens. & imag part of ac sens."),
 OPU("sens_mag", BJT2_QUEST_SENS_MAG, IF_REAL,   "sensitivity of ac magnitude"),
 OPU("sens_ph",   BJT2_QUEST_SENS_PH,   IF_REAL,    "sensitivity of ac phase"),
 OPU("sens_cplx", BJT2_QUEST_SENS_CPLX, IF_COMPLEX, "ac sensitivity"),
 IOPU("temp",     BJT2_TEMP,            IF_REAL,    "instance temperature"),
 IOPU("dtemp",    BJT2_DTEMP,           IF_REAL,    "instance temperature delta from circuit")
};

IFparm BJT2mPTable[] = { /* model parameters */
 OP("type",  BJT2_MOD_TYPE,  IF_STRING, "NPN or PNP"),
 IOPU("npn",  BJT2_MOD_NPN,  IF_FLAG, "NPN type device"),
 IOPU("pnp",  BJT2_MOD_PNP,  IF_FLAG, "PNP type device"),
 IOPU("subs", BJT2_MOD_SUBS, IF_INTEGER, "Vertical or Lateral device"),
 IOP("is",   BJT2_MOD_IS,   IF_REAL, "Saturation Current"),
 IOP("iss",   BJT2_MOD_ISS,   IF_REAL, "Substrate Jct. Saturation Current"),
 IOP("bf",   BJT2_MOD_BF,   IF_REAL, "Ideal forward beta"),
 IOP("nf",   BJT2_MOD_NF,   IF_REAL, "Forward emission coefficient"),
 IOP("vaf",  BJT2_MOD_VAF,  IF_REAL, "Forward Early voltage"),
 IOPR("va",  BJT2_MOD_VAF,  IF_REAL, "Forward Early voltage"),
 IOP("ikf",  BJT2_MOD_IKF,  IF_REAL, "Forward beta roll-off corner current"),
 IOPR("ik",  BJT2_MOD_IKF,  IF_REAL, "Forward beta roll-off corner current"),
 IOP("ise",  BJT2_MOD_ISE,  IF_REAL, "B-E leakage saturation current"),
#ifdef BJT2_COMPAT 
 IOP("c2",   BJT2_MOD_C2,   IF_REAL, "Obsolete parameter name"),
#endif 
 IOP("ne",   BJT2_MOD_NE,   IF_REAL, "B-E leakage emission coefficient"),
 IOP("br",   BJT2_MOD_BR,   IF_REAL, "Ideal reverse beta"),
 IOP("nr",   BJT2_MOD_NR,   IF_REAL, "Reverse emission coefficient"),
 IOP("var",  BJT2_MOD_VAR,  IF_REAL, "Reverse Early voltage"),
 IOPR("vb",  BJT2_MOD_VAR,  IF_REAL, "Reverse Early voltage"),
 IOP("ikr",  BJT2_MOD_IKR,  IF_REAL, "reverse beta roll-off corner current"),
 IOP("isc",  BJT2_MOD_ISC,  IF_REAL, "B-C leakage saturation current"),
#ifdef BJT2_COMPAT 
 IOP("c4",   BJT2_MOD_C4,   IF_REAL, "Obsolete parameter name"),
#endif
 IOP("nc",   BJT2_MOD_NC,   IF_REAL, "B-C leakage emission coefficient"),
 IOP("rb",   BJT2_MOD_RB,   IF_REAL, "Zero bias base resistance"),
 IOP("irb",  BJT2_MOD_IRB,  IF_REAL, "Current for base resistance=(rb+rbm)/2"),
 IOP("rbm",  BJT2_MOD_RBM,  IF_REAL, "Minimum base resistance"),
 IOP("re",   BJT2_MOD_RE,   IF_REAL, "Emitter resistance"),
 IOP("rc",   BJT2_MOD_RC,   IF_REAL, "Collector resistance"),
 IOPA("cje", BJT2_MOD_CJE,  IF_REAL,"Zero bias B-E depletion capacitance"),
 IOPA("vje",  BJT2_MOD_VJE,  IF_REAL, "B-E built in potential"),
 IOPR("pe",  BJT2_MOD_VJE,  IF_REAL, "B-E built in potential"),
 IOPA("mje",  BJT2_MOD_MJE,  IF_REAL, "B-E junction grading coefficient"),
 IOPR("me",  BJT2_MOD_MJE,  IF_REAL, "B-E junction grading coefficient"),
 IOPA("tf",  BJT2_MOD_TF,   IF_REAL, "Ideal forward transit time"),
 IOPA("xtf",  BJT2_MOD_XTF,  IF_REAL, "Coefficient for bias dependence of TF"),
 IOPA("vtf",  BJT2_MOD_VTF,  IF_REAL, "Voltage giving VBC dependence of TF"),
 IOPA("itf",  BJT2_MOD_ITF,  IF_REAL, "High current dependence of TF"),
 IOPA("ptf",  BJT2_MOD_PTF,  IF_REAL, "Excess phase"),
 IOPA("cjc",  BJT2_MOD_CJC,  IF_REAL, "Zero bias B-C depletion capacitance"),
 IOPA("vjc",  BJT2_MOD_VJC,  IF_REAL, "B-C built in potential"),
 IOPR("pc",  BJT2_MOD_VJC,  IF_REAL, "B-C built in potential"),
 IOPA("mjc",  BJT2_MOD_MJC,  IF_REAL, "B-C junction grading coefficient"),
 IOPR("mc",  BJT2_MOD_MJC,  IF_REAL, "B-C junction grading coefficient"),
 IOPA("xcjc",BJT2_MOD_XCJC, IF_REAL, "Fraction of B-C cap to internal base"),
 IOPA("tr",  BJT2_MOD_TR,   IF_REAL, "Ideal reverse transit time"),
 IOPA("cjs", BJT2_MOD_CJS,  IF_REAL, "Zero bias Substrate capacitance"),
 IOPR("csub", BJT2_MOD_CJS,  IF_REAL, "Zero bias Substrate capacitance"),
 IOPA("vjs",  BJT2_MOD_VJS,  IF_REAL, "Substrate junction built in potential"),
 IOPR("ps",  BJT2_MOD_VJS,  IF_REAL, "Substrate junction built in potential"),
 IOPA("mjs",  BJT2_MOD_MJS,  IF_REAL, "Substrate junction grading coefficient"),
 IOPR("ms",  BJT2_MOD_MJS,  IF_REAL, "Substrate junction grading coefficient"),
 IOP("xtb",  BJT2_MOD_XTB,  IF_REAL, "Forward and reverse beta temp. exp."),
 IOP("eg",   BJT2_MOD_EG,   IF_REAL, "Energy gap for IS temp. dependency"),
 IOP("xti",  BJT2_MOD_XTI,  IF_REAL, "Temp. exponent for IS"),
 IOP("tre1",  BJT2_MOD_TRE1,  IF_REAL, "Temp. coefficient 1 for RE"),
 IOP("tre2",  BJT2_MOD_TRE2,  IF_REAL, "Temp. coefficient 2 for RE"),
 IOP("trc1",  BJT2_MOD_TRC1,  IF_REAL, "Temp. coefficient 1 for RC"),
 IOP("trc2",  BJT2_MOD_TRC2,  IF_REAL, "Temp. coefficient 2 for RC"),
 IOP("trb1",  BJT2_MOD_TRB1,  IF_REAL, "Temp. coefficient 1 for RB"),
 IOP("trb2",  BJT2_MOD_TRB2,  IF_REAL, "Temp. coefficient 2 for RB"),
 IOP("trbm1",  BJT2_MOD_TRBM1,  IF_REAL, "Temp. coefficient 1 for RBM"),
 IOP("trbm2",  BJT2_MOD_TRBM2,  IF_REAL, "Temp. coefficient 2 for RBM"),
 IOP("fc",   BJT2_MOD_FC,   IF_REAL, "Forward bias junction fit parameter"),
 OPU("invearlyvoltf",BJT2_MOD_INVEARLYF,IF_REAL,"Inverse early voltage:forward"),
 OPU("invearlyvoltr",BJT2_MOD_INVEARLYR,IF_REAL,"Inverse early voltage:reverse"),
 OPU("invrollofff",BJT2_MOD_INVROLLOFFF,  IF_REAL,"Inverse roll off - forward"),
 OPU("invrolloffr",BJT2_MOD_INVROLLOFFR,  IF_REAL,"Inverse roll off - reverse"),
 OPU("collectorconduct",BJT2_MOD_COLCONDUCT,IF_REAL,"Collector conductance"),
 OPU("emitterconduct", BJT2_MOD_EMITTERCONDUCT,IF_REAL, "Emitter conductance"),
 OPU("transtimevbcfact",BJT2_MOD_TRANSVBCFACT,IF_REAL,"Transit time VBC factor"),
 OPU("excessphasefactor",BJT2_MOD_EXCESSPHASEFACTOR,IF_REAL,
							"Excess phase fact."),
 IOP("tnom", BJT2_MOD_TNOM, IF_REAL, "Parameter measurement temperature"),
 IOP("kf",   BJT2_MOD_KF,   IF_REAL, "Flicker Noise Coefficient"),
 IOP("af",BJT2_MOD_AF,   IF_REAL,"Flicker Noise Exponent")
};

char *BJT2names[] = {
    "collector",
    "base",
    "emitter",
    "substrate"
};


int	BJT2nSize = NUMELEMS(BJT2names);
int	BJT2pTSize = NUMELEMS(BJT2pTable);
int	BJT2mPTSize = NUMELEMS(BJT2mPTable);
int	BJT2iSize = sizeof(BJT2instance);
int	BJT2mSize = sizeof(BJT2model);
