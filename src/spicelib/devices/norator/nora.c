/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "noradefs.h"
#include "ngspice/suffix.h"

IFparm NORApTable[] = { /* parameters */ 
 OPU("pos_node", NORA_POS_NODE, IF_INTEGER, "Positive node of source"),
 OPU("neg_node", NORA_NEG_NODE, IF_INTEGER, "Negative node of source"),
 IP("ic", NORA_IC, IF_REAL, "Initial condition of branch current"),
 OP("i",        NORA_CURRENT,       IF_REAL,        "Output current"),
 OP("v",        NORA_VOLTS,         IF_REAL,        "Output voltage"),
 OP("p",        NORA_POWER,         IF_REAL,        "Power"),
};

char *NORAnames[] = {
    "V+",
    "V-",
};

int	NORAnSize = NUMELEMS(NORAnames);
int	NORApTSize = NUMELEMS(NORApTable);
int	NORAmPTSize = 0;
int	NORAiSize = sizeof(NORAinstance);
int	NORAmSize = sizeof(NORAmodel);
