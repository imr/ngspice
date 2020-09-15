/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "balundefs.h"
#include "ngspice/suffix.h"

IFparm BALUNpTable[] = { /* parameters */ 
 OPU("pos_node", BALUN_POS_NODE, IF_INTEGER, "Positive node"),
 OPU("neg_node", BALUN_NEG_NODE, IF_INTEGER, "Negative node"),
 OPU("cm_node", BALUN_CM_NODE, IF_INTEGER, "Common-mode node"),
 OPU("diff_node", BALUN_DIFF_NODE, IF_INTEGER, "Differential node"),
 OP("p",        BALUN_POWER,         IF_REAL,        "Power"),
};

char *BALUNnames[] = {
    "Pos",
    "Neg",
    "CM",
    "Diff"
};

int	BALUNnSize = NUMELEMS(BALUNnames);
int	BALUNpTSize = NUMELEMS(BALUNpTable);
int	BALUNmPTSize = 0;
int	BALUNiSize = sizeof(BALUNinstance);
int	BALUNmSize = sizeof(BALUNmodel);
