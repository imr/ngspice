/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "asrcdefs.h"
#include "ngspice/suffix.h"


/* Arbitrary source */
IFparm ASRCpTable[] = { /* parameters */
    IP("i", ASRC_CURRENT, IF_PARSETREE, "Current source"),
    IP("v", ASRC_VOLTAGE, IF_PARSETREE, "Voltage source"),
    IOPZU("temp", ASRC_TEMP, IF_REAL, "Instance operating temperature"),
    IOPZ("dtemp", ASRC_DTEMP, IF_REAL, "Instance temperature difference with the rest of the circuit"),
    IOPU("tc1", ASRC_TC1, IF_REAL, "First order temp. coefficient"),
    IOPU("tc2", ASRC_TC2, IF_REAL, "Second order temp. coefficient"),
    IOPU("reciproctc", ASRC_RTC, IF_INTEGER, "Flag to calculate reciprocal temperature behaviour"),
    IOPU("m", ASRC_M, IF_REAL, "Multiplier"),
    IOPU("reciprocm", ASRC_RM, IF_INTEGER, "Flag to calculate reciprocal multiplier behaviour"),
    OP("i", ASRC_OUTPUTCURRENT, IF_REAL, "Current through source"),
    OP("v", ASRC_OUTPUTVOLTAGE, IF_REAL, "Voltage across source"),
    OP("pos_node", ASRC_POS_NODE, IF_INTEGER, "Positive Node"),
    OP("neg_node", ASRC_NEG_NODE, IF_INTEGER, "Negative Node")
};


char *ASRCnames[] = {
    "src+",
    "src-"
};


int ASRCnSize = NUMELEMS(ASRCnames);
int ASRCpTSize = NUMELEMS(ASRCpTable);
int ASRCmPTSize = 0;
int ASRCiSize = sizeof(ASRCinstance);
int ASRCmSize = sizeof(ASRCmodel);
