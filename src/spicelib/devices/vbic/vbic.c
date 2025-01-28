/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This file defines the VBIC data structures that are
 * available to the next level(s) up the calling hierarchy
 */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "vbicdefs.h"
#include "ngspice/suffix.h"

IFparm VBICpTable[] = { /* parameters */
 IOPU("area",    VBIC_AREA,           IF_REAL,    "Area factor"),
 IOPU("off",     VBIC_OFF,            IF_FLAG,    "Device initially off"),
 IP("ic",        VBIC_IC,             IF_REALVEC, "Initial condition vector"),
 IOPAU("icvbe",  VBIC_IC_VBE,         IF_REAL,    "Initial B-E voltage"),
 IOPAU("icvce",  VBIC_IC_VCE,         IF_REAL,    "Initial C-E voltage"),
 IOPU("temp",    VBIC_TEMP,           IF_REAL,    "Instance temperature"),
 IOPU("dtemp",   VBIC_DTEMP,          IF_REAL,    "Instance delta temperature"),
 IOPU("m",       VBIC_M,              IF_REAL,    "Multiplier"),

 OPU("collnode", VBIC_QUEST_COLLNODE, IF_INTEGER, "Number of collector node"),
 OPU("basenode", VBIC_QUEST_BASENODE, IF_INTEGER, "Number of base node"),
 OPU("emitnode", VBIC_QUEST_EMITNODE, IF_INTEGER, "Number of emitter node"),
 OPU("subsnode", VBIC_QUEST_SUBSNODE, IF_INTEGER, "Number of substrate node"),
 OPU("collCXnode",VBIC_QUEST_COLLCXNODE,IF_INTEGER, "Internal collector node"),
 OPU("collCInode",VBIC_QUEST_COLLCINODE,IF_INTEGER, "Internal collector node"),
 OPU("baseBXnode",VBIC_QUEST_BASEBXNODE,IF_INTEGER, "Internal base node"),
 OPU("baseBInode",VBIC_QUEST_BASEBINODE,IF_INTEGER, "Internal base node"),
 OPU("baseBPnode",VBIC_QUEST_BASEBPNODE,IF_INTEGER, "Internal base node"),
 OPU("emitEInode",VBIC_QUEST_EMITEINODE,IF_INTEGER, "Internal emitter node"),
 OPU("subsSInode",VBIC_QUEST_SUBSSINODE,IF_INTEGER, "Internal substrate node"),
 OPU("xf1node", VBIC_QUEST_XF1NODE, IF_INTEGER, "Internal phase node xf1"),
 OPU("xf2node", VBIC_QUEST_XF2NODE, IF_INTEGER, "Internal phase node xf2"),
 OP("vbe",   VBIC_QUEST_VBE,  IF_REAL, "B-E voltage"),
 OP("vbc",   VBIC_QUEST_VBC,  IF_REAL, "B-C voltage"),
 OP("ic",    VBIC_QUEST_CC,   IF_REAL, "Collector current"),
 OP("ib",    VBIC_QUEST_CB,   IF_REAL, "Base current"),
 OP("ie",    VBIC_QUEST_CE,   IF_REAL, "Emitter current"),
 OP("is",    VBIC_QUEST_CS,   IF_REAL, "Substrate current"),
 OP("gm",    VBIC_QUEST_GM,   IF_REAL, "Small signal transconductance dIc/dVbe"),
 OP("go",    VBIC_QUEST_GO,   IF_REAL, "Small signal output conductance dIc/dVbc"),
 OP("gpi",   VBIC_QUEST_GPI,  IF_REAL, "Small signal input conductance dIb/dVbe"),
 OP("gmu",   VBIC_QUEST_GMU,  IF_REAL, "Small signal conductance dIb/dVbc"),
 OP("gx",    VBIC_QUEST_GX,   IF_REAL, "Conductance from base to internal base"),
 OP("cbe",   VBIC_QUEST_CBE,  IF_REAL, "Internal base to emitter capacitance"),
 OP("cbex",  VBIC_QUEST_CBEX, IF_REAL, "External base to emitter capacitance"),
 OP("cbc",   VBIC_QUEST_CBC,  IF_REAL, "Internal base to collector capacitance"),
 OP("cbcx",  VBIC_QUEST_CBCX, IF_REAL, "External Base to collector capacitance"),
 OP("cbep",  VBIC_QUEST_CBEP, IF_REAL, "Parasitic Base to emitter capacitance"),
 OP("cbcp",  VBIC_QUEST_CBCP, IF_REAL, "Parasitic Base to collector capacitance"),
 OP("p",     VBIC_QUEST_POWER,IF_REAL, "Power dissipation"),
 OPU("geqcb",VBIC_QUEST_GEQCB,IF_REAL, "Internal C-B-base cap. equiv. cond."),
 OPU("geqbx",VBIC_QUEST_GEQBX,IF_REAL, "External C-B-base cap. equiv. cond."),
 OPU("qbe",  VBIC_QUEST_QBE,  IF_REAL, "Charge storage B-E junction"),
 OPU("cqbe", VBIC_QUEST_CQBE, IF_REAL, "Cap. due to charge storage in B-E jct."),
 OPU("qbc",  VBIC_QUEST_QBC,  IF_REAL, "Charge storage B-C junction"),
 OPU("cqbc", VBIC_QUEST_CQBC, IF_REAL, "Cap. due to charge storage in B-C jct."),
 OPU("qbx",  VBIC_QUEST_QBX,  IF_REAL, "Charge storage B-X junction"),
 OPU("cqbx", VBIC_QUEST_CQBX, IF_REAL, "Cap. due to charge storage in B-X jct.")
};

IFparm VBICmPTable[] = { /* model parameters */
 OP("type",   VBIC_MOD_TYPE,  IF_STRING, "NPN or PNP"),
 IOP("selft", VBIC_MOD_SELFT, IF_INTEGER, "0: self-heating off, 1: self-heating on"),
 IOPU("npn",  VBIC_MOD_NPN,   IF_FLAG, "NPN type device"),
 IOPU("pnp",  VBIC_MOD_PNP,   IF_FLAG, "PNP type device"),
 IOP("tnom",  VBIC_MOD_TNOM,  IF_REAL, "Parameter measurement temperature"),
 IOPR("tref",  VBIC_MOD_TNOM,  IF_REAL, "Parameter measurement temperature"),
 IOP("rcx",   VBIC_MOD_RCX,   IF_REAL, "Extrinsic coll resistance"),
 IOP("rci",   VBIC_MOD_RCI,   IF_REAL, "Intrinsic coll resistance"),
 IOP("vo",    VBIC_MOD_VO,    IF_REAL, "Epi drift saturation voltage"),
 IOPR("v0",    VBIC_MOD_VO,    IF_REAL, "Epi drift saturation voltage"),
 IOP("gamm",  VBIC_MOD_GAMM,  IF_REAL, "Epi doping parameter"),
 IOP("hrcf",  VBIC_MOD_HRCF,  IF_REAL, "High current RC factor"),
 IOP("rbx",   VBIC_MOD_RBX,   IF_REAL, "Extrinsic base resistance"),
 IOP("rbi",   VBIC_MOD_RBI,   IF_REAL, "Intrinsic base resistance"),
 IOP("re",    VBIC_MOD_RE,    IF_REAL, "Intrinsic emitter resistance"),
 IOP("rs",    VBIC_MOD_RS,    IF_REAL, "Intrinsic substrate resistance"),
 IOP("rbp",   VBIC_MOD_RBP,   IF_REAL, "Parasitic base resistance"),
 IOP("is",    VBIC_MOD_IS,    IF_REAL, "Transport saturation current"),
 IOP("nf",    VBIC_MOD_NF,    IF_REAL, "Forward emission coefficient"),
 IOP("nr",    VBIC_MOD_NR,    IF_REAL, "Reverse emission coefficient"),
 IOP("fc",    VBIC_MOD_FC,    IF_REAL, "Fwd bias depletion capacitance limit"),
 IOP("cbeo",  VBIC_MOD_CBEO,  IF_REAL, "Extrinsic B-E overlap capacitance"),
 IOPR("cbe0",  VBIC_MOD_CBEO,  IF_REAL, "Extrinsic B-E overlap capacitance"),
 IOP("cje",   VBIC_MOD_CJE,   IF_REAL, "Zero bias B-E depletion capacitance"),
 IOP("pe",    VBIC_MOD_PE,    IF_REAL, "B-E built in potential"),
 IOP("me",    VBIC_MOD_ME,    IF_REAL, "B-E junction grading coefficient"),
 IOP("aje",   VBIC_MOD_AJE,   IF_REAL, "B-E capacitance smoothing factor"),
 IOP("cbco",  VBIC_MOD_CBCO,  IF_REAL, "Extrinsic B-C overlap capacitance"),
 IOPR("cbc0",  VBIC_MOD_CBCO,  IF_REAL, "Extrinsic B-C overlap capacitance"),
 IOP("cjc",   VBIC_MOD_CJC,   IF_REAL, "Zero bias B-C depletion capacitance"),
 IOP("qco",   VBIC_MOD_QCO,   IF_REAL, "Epi charge parameter"),
 IOPR("qc0",   VBIC_MOD_QCO,   IF_REAL, "Epi charge parameter"),
 IOP("cjep",  VBIC_MOD_CJEP,  IF_REAL, "B-C extrinsic zero bias capacitance"),
 IOP("pc",    VBIC_MOD_PC,    IF_REAL, "B-C built in potential"),
 IOP("mc",    VBIC_MOD_MC,    IF_REAL, "B-C junction grading coefficient"),
 IOP("ajc",   VBIC_MOD_AJC,   IF_REAL, "B-C capacitance smoothing factor"),
 IOP("cjcp",  VBIC_MOD_CJCP,  IF_REAL, "Zero bias S-C capacitance"),
 IOP("ps",    VBIC_MOD_PS,    IF_REAL, "S-C junction built in potential"),
 IOP("ms",    VBIC_MOD_MS,    IF_REAL, "S-C junction grading coefficient"),
 IOP("ajs",   VBIC_MOD_AJS,   IF_REAL, "S-C capacitance smoothing factor"),
 IOP("ibei",  VBIC_MOD_IBEI,  IF_REAL, "Ideal B-E saturation current"),
 IOP("wbe",   VBIC_MOD_WBE,   IF_REAL, "Portion of IBEI from Vbei, 1-WBE from Vbex"),
 IOP("nei",   VBIC_MOD_NEI,   IF_REAL, "Ideal B-E emission coefficient"),
 IOP("iben",  VBIC_MOD_IBEN,  IF_REAL, "Non-ideal B-E saturation current"),
 IOP("nen",   VBIC_MOD_NEN,   IF_REAL, "Non-ideal B-E emission coefficient"),
 IOP("ibci",  VBIC_MOD_IBCI,  IF_REAL, "Ideal B-C saturation current"),
 IOP("nci",   VBIC_MOD_NCI,   IF_REAL, "Ideal B-C emission coefficient"),
 IOP("ibcn",  VBIC_MOD_IBCN,  IF_REAL, "Non-ideal B-C saturation current"),
 IOP("ncn",   VBIC_MOD_NCN,   IF_REAL, "Non-ideal B-C emission coefficient"),
 IOP("avc1",  VBIC_MOD_AVC1,  IF_REAL, "B-C weak avalanche parameter 1"),
 IOP("avc2",  VBIC_MOD_AVC2,  IF_REAL, "B-C weak avalanche parameter 2"),
 IOP("isp",   VBIC_MOD_ISP,   IF_REAL, "Parasitic transport saturation current"),
 IOP("wsp",   VBIC_MOD_WSP,   IF_REAL, "Portion of ICCP"),
 IOP("nfp",   VBIC_MOD_NFP,   IF_REAL, "Parasitic fwd emission coefficient"),
 IOP("ibeip", VBIC_MOD_IBEIP, IF_REAL, "Ideal parasitic B-E saturation current"),
 IOP("ibenp", VBIC_MOD_IBENP, IF_REAL, "Non-ideal parasitic B-E saturation current"),
 IOP("ibcip", VBIC_MOD_IBCIP, IF_REAL, "Ideal parasitic B-C saturation current"),
 IOP("ncip",  VBIC_MOD_NCIP,  IF_REAL, "Ideal parasitic B-C emission coefficient"),
 IOP("ibcnp", VBIC_MOD_IBCNP, IF_REAL, "Nonideal parasitic B-C saturation current"),
 IOP("ncnp",  VBIC_MOD_NCNP,  IF_REAL, "Nonideal parasitic B-C emission coefficient"),
 IOP("vef",   VBIC_MOD_VEF,   IF_REAL, "Forward Early voltage"),
 IOP("ver",   VBIC_MOD_VER,   IF_REAL, "Reverse Early voltage"),
 IOP("ikf",   VBIC_MOD_IKF,   IF_REAL, "Forward knee current"),
 IOP("ikr",   VBIC_MOD_IKR,   IF_REAL, "Reverse knee current"),
 IOP("ikp",   VBIC_MOD_IKP,   IF_REAL, "Parasitic knee current"),
 IOP("tf",    VBIC_MOD_TF,    IF_REAL, "Ideal forward transit time"),
 IOP("qtf",   VBIC_MOD_QTF,   IF_REAL, "Variation of TF with base-width modulation"),
 IOP("xtf",   VBIC_MOD_XTF,   IF_REAL, "Coefficient for bias dependence of TF"),
 IOP("vtf",   VBIC_MOD_VTF,   IF_REAL, "Voltage giving VBC dependence of TF"),
 IOP("itf",   VBIC_MOD_ITF,   IF_REAL, "High current dependence of TF"),
 IOP("tr",    VBIC_MOD_TR,    IF_REAL, "Ideal reverse transit time"),
 IOP("td",    VBIC_MOD_TD,    IF_REAL, "Forward excess-phase delay time"),
 IOP("kfn",   VBIC_MOD_KFN,   IF_REAL, "B-E Flicker Noise Coefficient"),
 IOP("afn",   VBIC_MOD_AFN,   IF_REAL, "B-E Flicker Noise Exponent"),
 IOP("bfn",   VBIC_MOD_BFN,   IF_REAL, "B-E Flicker Noise 1/f dependence"),
 IOP("xre",   VBIC_MOD_XRE,   IF_REAL, "Temperature exponent of RE"),
 IOP("xrbi",  VBIC_MOD_XRBI,  IF_REAL, "Temperature exponent of RBI"),
 IOP("xrci",  VBIC_MOD_XRCI,  IF_REAL, "Temperature exponent of RCI"),
 IOP("xrs",   VBIC_MOD_XRS,   IF_REAL, "Temperature exponent of RS"),
 IOP("xvo",   VBIC_MOD_XVO,   IF_REAL, "Temperature exponent of VO"),
 IOPR("xv0",   VBIC_MOD_XVO,   IF_REAL, "Temperature exponent of VO"),
 IOP("ea",    VBIC_MOD_EA,    IF_REAL, "Activation energy for IS"),
 IOP("eaie",  VBIC_MOD_EAIE,  IF_REAL, "Activation energy for IBEI"),
 IOP("eaic",  VBIC_MOD_EAIC,  IF_REAL, "Activation energy for IBCI/IBEIP"),
 IOP("eais",  VBIC_MOD_EAIS,  IF_REAL, "Activation energy for IBCIP"),
 IOP("eane",  VBIC_MOD_EANE,  IF_REAL, "Activation energy for IBEN"),
 IOP("eanc",  VBIC_MOD_EANC,  IF_REAL, "Activation energy for IBCN/IBENP"),
 IOP("eans",  VBIC_MOD_EANS,  IF_REAL, "Activation energy for IBCNP"),
 IOP("xis",   VBIC_MOD_XIS,   IF_REAL, "Temperature exponent of IS"),
 IOP("xii",   VBIC_MOD_XII,   IF_REAL, "Temperature exponent of IBEI,IBCI,IBEIP,IBCIP"),
 IOP("xin",   VBIC_MOD_XIN,   IF_REAL, "Temperature exponent of IBEN,IBCN,IBENP,IBCNP"),
 IOP("tnf",   VBIC_MOD_TNF,   IF_REAL, "Temperature exponent of NF"),
 IOP("tavc",  VBIC_MOD_TAVC,  IF_REAL, "Temperature exponent of AVC2"),
 IOP("rth",   VBIC_MOD_RTH,   IF_REAL, "Thermal resistance"),
 IOP("cth",   VBIC_MOD_CTH,   IF_REAL, "Thermal capacitance"),
 IOP("vrt",   VBIC_MOD_VRT,   IF_REAL, "Punch-through voltage of internal B-C junction"),
 IOP("art",   VBIC_MOD_ART,   IF_REAL, "Smoothing parameter for reach-through"),
 IOP("ccso",  VBIC_MOD_CCSO,  IF_REAL, "Fixed C-S capacitance"),
 IOP("qbm",   VBIC_MOD_QBM,   IF_REAL, "Select SGP qb formulation"),
 IOP("nkf",   VBIC_MOD_NKF,   IF_REAL, "High current beta rolloff"),
 IOP("xikf",  VBIC_MOD_XIKF,  IF_REAL, "Temperature exponent of IKF"),
 IOP("xrcx",  VBIC_MOD_XRCX,  IF_REAL, "Temperature exponent of RCX"),
 IOPR("xrc",  VBIC_MOD_XRCX,  IF_REAL, "Temperature exponent of RCX"),
 IOP("xrbx",  VBIC_MOD_XRBX,  IF_REAL, "Temperature exponent of RBX"),
 IOPR("xrb",  VBIC_MOD_XRBX,  IF_REAL, "Temperature exponent of RBX"),
 IOP("xrbp",  VBIC_MOD_XRBP,  IF_REAL, "Temperature exponent of RBP"),
 IOP("isrr",  VBIC_MOD_ISRR,  IF_REAL, "Separate IS for fwd and rev"),
 IOP("xisr",  VBIC_MOD_XISR,  IF_REAL, "Temperature exponent of ISR"),
 IOP("dear",  VBIC_MOD_DEAR,  IF_REAL, "Delta activation energy for ISRR"),
 IOP("eap",   VBIC_MOD_EAP,   IF_REAL, "Exitivation energy for ISP"),
 IOP("vbbe",  VBIC_MOD_VBBE,  IF_REAL, "B-E breakdown voltage"),
 IOP("nbbe",  VBIC_MOD_NBBE,  IF_REAL, "B-E breakdown emission coefficient"),
 IOP("ibbe",  VBIC_MOD_IBBE,  IF_REAL, "B-E breakdown current"),
 IOP("tvbbe1",VBIC_MOD_TVBBE1,IF_REAL, "Linear temperature coefficient of VBBE"),
 IOP("tvbbe2",VBIC_MOD_TVBBE2,IF_REAL, "Quadratic temperature coefficient of VBBE"),
 IOP("tnbbe", VBIC_MOD_TNBBE, IF_REAL, "Temperature coefficient of NBBE"),
 IOP("ebbe",  VBIC_MOD_EBBE,  IF_REAL, "exp(-VBBE/(NBBE*Vtv))"),
 IOP("dtemp", VBIC_MOD_DTEMP, IF_REAL, "Locale Temperature difference"),
 IOP("vers",  VBIC_MOD_VERS,  IF_REAL, "Revision Version"),
 IOP("vref",  VBIC_MOD_VREF,  IF_REAL, "Reference Version"),
 IOP("vbe_max", VBIC_MOD_VBE_MAX, IF_REAL, "maximum voltage B-E junction"),
 IOPR("bvbe", VBIC_MOD_VBE_MAX, IF_REAL, "maximum voltage B-E junction"),
 IOP("vbc_max", VBIC_MOD_VBC_MAX, IF_REAL, "maximum voltage B-C junction"),
 IOPR("bvbc", VBIC_MOD_VBC_MAX, IF_REAL, "maximum voltage B-C junction"),
 IOP("vce_max", VBIC_MOD_VCE_MAX, IF_REAL, "maximum voltage C-E branch"),
 IOPR("bvce", VBIC_MOD_VCE_MAX, IF_REAL, "maximum voltage C-E branch"),
 IOP("vsub_max", VBIC_MOD_VSUB_MAX, IF_REAL, "maximum voltage C-substrate branch"),
 IOPR("bvsub", VBIC_MOD_VSUB_MAX, IF_REAL, "maximum voltage C-substrate branch"),
 IOP("vbefwd", VBIC_MOD_VBEFWD_MAX, IF_REAL, "maximum forward voltage B-E junction"),
 IOP("vbcfwd", VBIC_MOD_VBCFWD_MAX, IF_REAL, "maximum forward voltage B-C junction"),
 IOP("vsubfwd", VBIC_MOD_VSUBFWD_MAX, IF_REAL, "maximum forward voltage C-substrate junction")
};

char *VBICnames[] = {
    "collector",
    "base",
    "emitter",
    "substrate",
    "temp"
};


int VBICnSize = NUMELEMS(VBICnames);
int VBICpTSize = NUMELEMS(VBICpTable);
int VBICmPTSize = NUMELEMS(VBICmPTable);
int VBICiSize = sizeof(VBICinstance);
int VBICmSize = sizeof(VBICmodel);
