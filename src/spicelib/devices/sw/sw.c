/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "swdefs.h"
#include "ngspice/suffix.h"

IFparm SWpTable[] = { /* parameters */
    IP("on",           SW_IC_ON,         IF_FLAG,    "Switch initially closed"),
    IP("off",          SW_IC_OFF,        IF_FLAG,    "Switch initially open"),
    IOPU("pos_node",   SW_POS_NODE,      IF_INTEGER, "Positive node of switch"),
    IOPU("neg_node",   SW_NEG_NODE,      IF_INTEGER, "Negative node of switch"),
    OPU("cont_p_node", SW_POS_CONT_NODE, IF_INTEGER, "Positive contr. node of switch"),
    OPU("cont_n_node", SW_NEG_CONT_NODE, IF_INTEGER, "Positive contr. node of switch"),
    OP("i",            SW_CURRENT,       IF_REAL,    "Switch current"),
    OP("p",            SW_POWER,         IF_REAL,    "Switch power")
};

IFparm SWmPTable[] = { /* model parameters */
    IOPU("sw",   SW_MOD_SW,   IF_FLAG, "Switch model"),
    IOPU("vt",   SW_MOD_VTH,  IF_REAL, "Threshold voltage"),
    IOPU("vh",   SW_MOD_VHYS, IF_REAL, "Hysteresis voltage"),
    IOPU("ron",  SW_MOD_RON,  IF_REAL, "Resistance when closed"),
    OPU("gon",   SW_MOD_GON,  IF_REAL, "Conductance when closed"),
    IOPU("roff", SW_MOD_ROFF, IF_REAL, "Resistance when open"),
    OPU("goff",  SW_MOD_GOFF, IF_REAL, "Conductance when open")
};

char *SWnames[] = {
    "S+",
    "S-",
    "SC+",
    "SC-"
};

int SWnSize = NUMELEMS(SWnames);
int SWpTSize = NUMELEMS(SWpTable);
int SWmPTSize = NUMELEMS(SWmPTable);
int SWiSize = sizeof(SWinstance);
int SWmSize = sizeof(SWmodel);
