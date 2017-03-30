/**********
Based on jfet.c 
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994:  Parameter definitions called from jfetparm.h
                 Extra state vectors added to JFET2pTable
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "jfet2defs.h"
#include "ngspice/suffix.h"

IFparm JFET2pTable[] = { /* device parameters */ 
 IOPU("off",         JFET2_OFF,            IF_FLAG,   "Device initially off"),
 IOPAU("ic",         JFET2_IC,             IF_REALVEC,"Initial VDS,VGS vector"),
 IOPU("area",        JFET2_AREA,           IF_REAL,   "Area factor"),
 IOPU("m",           JFET2_M,              IF_REAL,   "Parallel Multiplier"),
 IOPAU("ic-vds",     JFET2_IC_VDS,         IF_REAL,   "Initial D-S voltage"),
 IOPAU("ic-vgs",     JFET2_IC_VGS,         IF_REAL,   "Initial G-S volrage"),
 IOPU("temp",        JFET2_TEMP,           IF_REAL,   "Instance temperature"),
 IOPU("dtemp",       JFET2_DTEMP,          IF_REAL,   "Instance temperature difference"),
 OPU("drain-node",   JFET2_DRAINNODE,      IF_INTEGER,"Number of drain node"),
 OPU("gate-node",    JFET2_GATENODE,       IF_INTEGER,"Number of gate node"),
 OPU("source-node",  JFET2_SOURCENODE,     IF_INTEGER,"Number of source node"),
 OPU("drain-prime-node", JFET2_DRAINPRIMENODE, IF_INTEGER,"Internal drain node"),
 OPU("source-prime-node",JFET2_SOURCEPRIMENODE,IF_INTEGER,"Internal source node"),
 OP("vgs",   JFET2_VGS,  IF_REAL, "Voltage G-S"),
 OP("vgd",   JFET2_VGD,  IF_REAL, "Voltage G-D"),
 OP("ig",    JFET2_CG,   IF_REAL, "Current at gate node"),
 OP("id",    JFET2_CD,   IF_REAL, "Current at drain node"),
 OP("is",    JFET2_CS,   IF_REAL, "Source current"),
 OP("igd",   JFET2_CGD,  IF_REAL, "Current G-D"),
 OP("gm",    JFET2_GM,   IF_REAL, "Transconductance"),
 OP("gds",   JFET2_GDS,  IF_REAL, "Conductance D-S"),
 OP("ggs",   JFET2_GGS,  IF_REAL, "Conductance G-S"),
 OP("ggd",   JFET2_GGD,  IF_REAL, "Conductance G-D"),
 OPU("qgs",  JFET2_QGS,  IF_REAL, "Charge storage G-S junction"),
 OPU("qgd",  JFET2_QGD,  IF_REAL, "Charge storage G-D junction"),
 OPU("cqgs", JFET2_CQGS, IF_REAL, "Capacitance due to charge storage G-S junction"),
 OPU("cqgd", JFET2_CQGD, IF_REAL, "Capacitance due to charge storage G-D junction"),
 OPU("p",    JFET2_POWER,IF_REAL, "Power dissipated by the JFET2"),
 OPU("vtrap",JFET2_VTRAP,IF_REAL, "Quiescent drain feedback potential"),
 OPU("vpave",JFET2_PAVE, IF_REAL, "Quiescent power dissipation"),
};

IFparm JFET2mPTable[] = { /* model parameters */
 OP("type",     JFET2_MOD_TYPE,    IF_STRING, "N-type or P-type JFET2 model"),
 IOP("njf",     JFET2_MOD_NJF,     IF_FLAG,"N type JFET2 model"),
 IOP("pjf",     JFET2_MOD_PJF,     IF_FLAG,"P type JFET2 model"),
#define  PARAM(code,id,flag,ref,default,descrip) IOP(code,id,IF_REAL,descrip),
#define PARAMR(code,id,flag,ref,default,descrip) IOPR(code,id,IF_REAL,descrip),
#define PARAMA(code,id,flag,ref,default,descrip) IOPA(code,id,IF_REAL,descrip),
#include "jfet2parm.h"

 OPU("gd", JFET2_MOD_DRAINCONDUCT, IF_REAL,"Drain conductance"),
 OPU("gs", JFET2_MOD_SOURCECONDUCT,IF_REAL,"Source conductance"),
 IOPU("tnom",   JFET2_MOD_TNOM,    IF_REAL,"parameter measurement temperature"),
};


char *JFET2names[] = {
    "Drain",
    "Gate",
    "Source"
};

int	JFET2nSize = NUMELEMS(JFET2names);
int	JFET2pTSize = NUMELEMS(JFET2pTable);
int	JFET2mPTSize = NUMELEMS(JFET2mPTable);
int	JFET2iSize = sizeof(JFET2instance);
int	JFET2mSize = sizeof(JFET2model);
