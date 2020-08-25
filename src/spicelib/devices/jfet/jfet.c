/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
	Laboratory for Communication Science Engineering
	Sydney University Department of Electrical Engineering, Australia
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "jfetdefs.h"
#include "ngspice/suffix.h"

IFparm JFETpTable[] = { /* device parameters */ 
 IOPU("off",         JFET_OFF,            IF_FLAG,   "Device initially off"),
 IOPAU("ic",          JFET_IC,             IF_REALVEC,"Initial VDS,VGS vector"),
 IOPU("area",        JFET_AREA,           IF_REAL,   "Area factor"),
 IOPU("m",           JFET_M,              IF_REAL,   "Parallel multiplier"),
 IOPAU("ic-vds",      JFET_IC_VDS,         IF_REAL,   "Initial D-S voltage"),
 IOPAU("ic-vgs",      JFET_IC_VGS,         IF_REAL,   "Initial G-S volrage"),
 IOPU("temp",        JFET_TEMP,           IF_REAL,   "Instance temperature"),
 IOPU("dtemp",        JFET_DTEMP,           IF_REAL,   "Instance temperature difference"),
 OPU("drain-node",   JFET_DRAINNODE,      IF_INTEGER,"Number of drain node"),
 OPU("gate-node",    JFET_GATENODE,       IF_INTEGER,"Number of gate node"),
 OPU("source-node",  JFET_SOURCENODE,     IF_INTEGER,"Number of source node"),
 OPU("drain-prime-node", JFET_DRAINPRIMENODE, IF_INTEGER,"Internal drain node"),
 OPU("source-prime-node",JFET_SOURCEPRIMENODE,IF_INTEGER,
							"Internal source node"),
 OP("vgs",          JFET_VGS,            IF_REAL,   "Voltage G-S"),
 OP("vgd",          JFET_VGD,            IF_REAL,   "Voltage G-D"),
 OP("ig",           JFET_CG,             IF_REAL,   "Current at gate node"),
 OP("id",           JFET_CD,             IF_REAL,   "Current at drain node"),
 OP("is",  	    JFET_CS,   		 IF_REAL,   "Source current"),
 OP("igd",          JFET_CGD,            IF_REAL,   "Current G-D"),
 OP("gm",           JFET_GM,             IF_REAL,   "Transconductance"),
 OP("gds",          JFET_GDS,            IF_REAL,   "Conductance D-S"),
 OP("ggs",          JFET_GGS,            IF_REAL,   "Conductance G-S"),
 OP("ggd",          JFET_GGD,            IF_REAL,   "Conductance G-D"),
 OPU("qgs", JFET_QGS,  IF_REAL,"Charge storage G-S junction"),
 OPU("qgd", JFET_QGD,  IF_REAL,"Charge storage G-D junction"),
 OPU("cqgs",JFET_CQGS, IF_REAL,
			"Capacitance due to charge storage G-S junction"),
 OPU("cqgd",JFET_CQGD, IF_REAL,
			"Capacitance due to charge storage G-D junction"),
 OPU("p",   JFET_POWER,IF_REAL,"Power dissipated by the JFET"),
};

IFparm JFETmPTable[] = { /* model parameters */
 OP("type",     JFET_MOD_TYPE,    IF_STRING, "N-type or P-type JFET model"),
 IP("njf",     JFET_MOD_NJF,     IF_FLAG,"N type JFET model"),
 IP("pjf",     JFET_MOD_PJF,     IF_FLAG,"P type JFET model"),
 IOP("vt0",     JFET_MOD_VTO,     IF_REAL,"Threshold voltage"),
 IOPR("vto",     JFET_MOD_VTO,    IF_REAL,"Threshold voltage"),
 IOP("beta",    JFET_MOD_BETA,    IF_REAL,"Transconductance parameter"),
 IOP("lambda",  JFET_MOD_LAMBDA,  IF_REAL,"Channel length modulation param."),
 IOP("rd",      JFET_MOD_RD,      IF_REAL,"Drain ohmic resistance"),
 OPU("gd", JFET_MOD_DRAINCONDUCT, IF_REAL,"Drain conductance"),
 IOP("rs",      JFET_MOD_RS,      IF_REAL,"Source ohmic resistance"),
 OPU("gs",JFET_MOD_SOURCECONDUCT,IF_REAL,"Source conductance"),
 IOPA("cgs",     JFET_MOD_CGS,    IF_REAL,"G-S junction capactance"),
 IOPA("cgd",     JFET_MOD_CGD,    IF_REAL,"G-D junction cap"),
 IOP("pb",      JFET_MOD_PB,      IF_REAL,"Gate junction potential"),
 IOP("is",      JFET_MOD_IS,      IF_REAL,"Gate junction saturation current"),
 IOP("fc",      JFET_MOD_FC,      IF_REAL,"Forward bias junction fit parm."),
 /* Modification for Sydney University JFET model */
 IOP("b",     JFET_MOD_B,        IF_REAL,"Doping tail parameter"),
 /* end Sydney University mod. */
 IOPU("tnom",   JFET_MOD_TNOM,    IF_REAL,"parameter measurement temperature"),
 IOP("tcv", JFET_MOD_TCV, IF_REAL, "Threshold voltage temperature coefficient"),
 IOP("vtotc", JFET_MOD_VTOTC, IF_REAL, "Threshold voltage temperature coefficient alternative"),
 IOP("bex", JFET_MOD_BEX, IF_REAL, "Mobility temperature exponent"),
 IOP("betatce", JFET_MOD_BETATCE, IF_REAL, "Mobility temperature exponent alternative"),
 IOP("xti", JFET_MOD_XTI, IF_REAL, "Gate junction saturation current temperature exponent"),
 IOP("eg",  JFET_MOD_EG, IF_REAL, "Bandgap voltage"),
 IOP("kf",  JFET_MOD_KF, IF_REAL, "Flicker Noise Coefficient"),
 IOP("af",  JFET_MOD_AF, IF_REAL, "Flicker Noise Exponent"),
 IOP("nlev",JFET_MOD_NLEV, IF_INTEGER, "Noise equation selector"),
 IOP("gdsnoi", JFET_MOD_GDSNOI, IF_REAL, "Channel noise coefficient")
};


char *JFETnames[] = {
    "Drain",
    "Gate",
    "Source"
};

int	JFETnSize = NUMELEMS(JFETnames);
int	JFETpTSize = NUMELEMS(JFETpTable);
int	JFETmPTSize = NUMELEMS(JFETmPTable);
int	JFETiSize = sizeof(JFETinstance);
int	JFETmSize = sizeof(JFETmodel);
