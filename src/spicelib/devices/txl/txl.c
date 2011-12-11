/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "txldefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"

IFparm TXLpTable[] = { 
	IP("pos_node",   TXL_IN_NODE,     IF_INTEGER,"Positive node of txl"),
	IP("neg_node",   TXL_OUT_NODE,    IF_INTEGER,"Negative node of txl"),
 	IOP("length",    TXL_LENGTH,      IF_REAL,"length of line"),
};

IFparm TXLmPTable[] = { /* model parameters */
 IOP( "r",         TXL_R,           IF_REAL,"resistance per length"),
 IOP( "l",         TXL_L,           IF_REAL,"inductance per length"),
 IOP( "c",         TXL_C,           IF_REAL,"capacitance per length"),
 IOP( "g",         TXL_G,           IF_REAL,"conductance per length"),
 IOP( "length",    TXL_length,      IF_REAL,"length"),
 IP( "txl",      TXL_MOD_R,         IF_FLAG,"Device is a txl model"),
};

char *TXLnames[] = {
	"Y+",
	"Y-"
};

int TXLnSize = NUMELEMS(TXLnames);
int TXLiSize = sizeof(TXLinstance);
int TXLmSize = sizeof(TXLmodel);
int TXLmPTSize = NUMELEMS(TXLmPTable);
int TXLpTSize = NUMELEMS(TXLpTable);
