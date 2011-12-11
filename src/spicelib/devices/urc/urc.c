/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "urcdefs.h"
#include "ngspice/suffix.h"

IFparm URCpTable[] = { /* parameters */ 
 IOPU( "l",      URC_LEN,   IF_REAL, "Length of transmission line"),
 IOPU( "n",      URC_LUMPS, IF_INTEGER, "Number of lumps"),
 OPU( "pos_node",URC_POS_NODE,IF_INTEGER,"Positive node of URC"),
 OPU( "neg_node",URC_NEG_NODE,IF_INTEGER,"Negative node of URC"),
 OPU( "gnd",     URC_GND_NODE,IF_INTEGER,"Ground node of URC")
};

IFparm URCmPTable[] = { /* model parameters */
 IOP( "k",      URC_MOD_K,      IF_REAL, "Propagation constant"),
 IOPA( "fmax",   URC_MOD_FMAX,   IF_REAL, "Maximum frequency of interest"),
 IOP( "rperl",  URC_MOD_RPERL,  IF_REAL, "Resistance per unit length"),
 IOPA( "cperl",  URC_MOD_CPERL,  IF_REAL, "Capacitance per unit length"),
 IOP( "isperl", URC_MOD_ISPERL, IF_REAL, "Saturation current per length"),
 IOP( "rsperl", URC_MOD_RSPERL, IF_REAL, "Diode resistance per length"),
 IP( "urc",    URC_MOD_URC,    IF_FLAG, "Uniform R.C. line model")
};

char *URCnames[] = {
    "P1",
    "P2",
    "Ref"
};

int	URCnSize = NUMELEMS(URCnames);
int	URCpTSize = NUMELEMS(URCpTable);
int	URCmPTSize = NUMELEMS(URCmPTable);
int	URCiSize = sizeof(URCinstance);
int	URCmSize = sizeof(URCmodel);
