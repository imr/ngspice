/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "mesdefs.h"
#include "ngspice/suffix.h"

IFparm MESpTable[] = { /* parameters */ 
 OPU("off",  MES_OFF,    IF_FLAG   ,"Device initially off"),
 IOPU("area", MES_AREA,   IF_REAL   ,"Area factor"),
 IOPUR("m",   MES_AREA,   IF_REAL   ,"Parallel Multiplier"),
 IOPAU("icvds",  MES_IC_VDS, IF_REAL   ,"Initial D-S voltage"),
 IOPAU("icvgs",  MES_IC_VGS, IF_REAL   ,"Initial G-S voltage"),
 OPU("dnode", MES_DRAINNODE, IF_INTEGER,"Number of drain node"),
 OPU("gnode", MES_GATENODE, IF_INTEGER,"Number of gate node"),
 OPU("snode", MES_SOURCENODE, IF_INTEGER,"Number of source node"),
 OPU("dprimenode",MES_DRAINPRIMENODE,IF_INTEGER,"Number of internal drain node"),
 OPU("sprimenode",MES_SOURCEPRIMENODE,IF_INTEGER,
        "Number of internal source node"),
 OP("vgs", MES_VGS, IF_REAL,"Gate-Source voltage"),
 OP("vgd", MES_VGD, IF_REAL,"Gate-Drain voltage"),
 OP("cg", MES_CG, IF_REAL,"Gate capacitance"),
 OP("cd", MES_CD, IF_REAL,"Drain capacitance"),
 OP("cgd", MES_CGD, IF_REAL,"Gate-Drain capacitance"),
 OP("gm", MES_GM, IF_REAL,"Transconductance"),
 OP("gds", MES_GDS, IF_REAL,"Drain-Source conductance"),
 OP("ggs", MES_GGS, IF_REAL,"Gate-Source conductance"),
 OP("ggd", MES_GGD, IF_REAL,"Gate-Drain conductance"),
 OP("cqgs", MES_CQGS, IF_REAL,"Capacitance due to gate-source charge storage"),
 OP("cqgd", MES_CQGD, IF_REAL,"Capacitance due to gate-drain charge storage"),
 OPU("qgs", MES_QGS, IF_REAL,"Gate-Source charge storage"),
 OPU("qgd", MES_QGD, IF_REAL,"Gate-Drain charge storage"),
 OP("is",   MES_CS,     IF_REAL   ,"Source current"),
 OP("p",    MES_POWER,  IF_REAL   ,"Power dissipated by the mesfet")
};

IFparm MESmPTable[] = { /* model parameters */
 OP( "type",    MES_MOD_TYPE,    IF_FLAG,"N-type or P-type MESfet model"),
 IP( "nmf",    MES_MOD_NMF,    IF_FLAG,"N type MESfet model"),
 IP( "pmf",    MES_MOD_PMF,    IF_FLAG,"P type MESfet model"),
 IOP( "vt0",    MES_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOPR( "vto",    MES_MOD_VTO,    IF_REAL,"Pinch-off voltage"),
 IOP( "alpha",  MES_MOD_ALPHA,  IF_REAL,"Saturation voltage parameter"),
 IOP( "beta",   MES_MOD_BETA,   IF_REAL,"Transconductance parameter"),
 IOP( "lambda", MES_MOD_LAMBDA, IF_REAL,"Channel length modulation parm."),
 IOP( "b",      MES_MOD_B,      IF_REAL,"Doping tail extending parameter"),
 IOP( "rd",     MES_MOD_RD,     IF_REAL,"Drain ohmic resistance"),
 OPU( "gd",  MES_MOD_DRAINCONDUCT,   IF_REAL,"Drain conductance"),
 IOP( "rs",     MES_MOD_RS,     IF_REAL,"Source ohmic resistance"),
 OPU( "gs",  MES_MOD_SOURCECONDUCT,  IF_REAL,"Source conductance"),
 IOPA( "cgs",    MES_MOD_CGS,    IF_REAL,"G-S junction capacitance"),
 IOPA( "cgd",    MES_MOD_CGD,    IF_REAL,"G-D junction capacitance"),
 IOP( "pb",     MES_MOD_PB,     IF_REAL,"Gate junction potential"),
 IOP( "is",     MES_MOD_IS,     IF_REAL,"Junction saturation current"),
 IOP( "fc",     MES_MOD_FC,     IF_REAL,"Forward bias junction fit parm."),
 OP( "depl_cap",    MES_MOD_DEPLETIONCAP,   IF_REAL,"Depletion capacitance"),
 OP( "vcrit",   MES_MOD_VCRIT,  IF_REAL,"Critical voltage"),
 IOP("kf",        MES_MOD_KF,           IF_REAL,"Flicker noise coefficient"),
 IOP("af",        MES_MOD_AF,           IF_REAL,"Flicker noise exponent")
};

char *MESnames[] = {
    "Drain",
    "Gate",
    "Source"
};

int	MESnSize = NUMELEMS(MESnames);
int	MESpTSize = NUMELEMS(MESpTable);
int	MESmPTSize = NUMELEMS(MESmPTable);
int	MESiSize = sizeof(MESinstance);
int	MESmSize = sizeof(MESmodel);
