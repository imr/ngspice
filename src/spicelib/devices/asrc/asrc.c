/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "devdefs.h"
#include "asrcdefs.h"
#include "suffix.h"

/* Arbitrary source */
IFparm ASRCpTable[] = { /* parameters */ 
 IP( "i", ASRC_CURRENT, IF_PARSETREE, "Current source "),
 IP( "v", ASRC_VOLTAGE, IF_PARSETREE, "Voltage source"),
 OP( "i", ASRC_OUTPUTCURRENT, IF_REAL, "Current through source "),
 OP( "v", ASRC_OUTPUTVOLTAGE, IF_REAL, "Voltage across source"),
 OP( "pos_node", ASRC_POS_NODE, IF_INTEGER, "Positive Node"),
 OP( "neg_node", ASRC_NEG_NODE, IF_INTEGER, "Negative Node")
};

char *ASRCnames[] = {
    "src+",
    "src-"
};

int	ASRCnSize = NUMELEMS(ASRCnames);
int	ASRCpTSize = NUMELEMS(ASRCpTable);
int	ASRCmPTSize = 0;
int	ASRCiSize = sizeof(ASRCinstance);
int	ASRCmSize = sizeof(ASRCmodel);
