/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "cswdefs.h"
#include "ngspice/suffix.h"


IFparm CSWpTable[] = { /* parameters */
    IOP("control",  CSW_CONTROL,  IF_INSTANCE, "Name of controlling source"),
    IP("on",        CSW_IC_ON,    IF_FLAG,     "Initially closed"),
    IP("off",       CSW_IC_OFF,   IF_FLAG,     "Initially open"),
    OPU("pos_node", CSW_POS_NODE, IF_INTEGER,  "Positive node of switch"),
    OPU("neg_node", CSW_NEG_NODE, IF_INTEGER,  "Negative node of switch"),
    OP("i",         CSW_CURRENT,  IF_REAL,     "Switch current"),
    OP("p",         CSW_POWER,    IF_REAL,     "Instantaneous power")
};

IFparm CSWmPTable[] = { /* model parameters */
    IOPU("csw",  CSW_CSW,  IF_FLAG, "Current controlled switch model"),
    IOPU("it",   CSW_ITH,  IF_REAL, "Threshold current"),
    IOPU("ih",   CSW_IHYS, IF_REAL, "Hysterisis current"),
    IOPU("ron",  CSW_RON,  IF_REAL, "Closed resistance"),
    IOPU("roff", CSW_ROFF, IF_REAL, "Open resistance"),
    OPU("gon",   CSW_GON,  IF_REAL, "Closed conductance"),
    OPU("goff",  CSW_GOFF, IF_REAL, "Open conductance")
};

char *CSWnames[] = {
    "W+",
    "W-"
};

int CSWnSize = NUMELEMS(CSWnames);
int CSWpTSize = NUMELEMS(CSWpTable);
int CSWmPTSize = NUMELEMS(CSWmPTable);
int CSWiSize = sizeof(CSWinstance);
int CSWmSize = sizeof(CSWmodel);
