/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice.h"
#include "cpldefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "suffix.h"

IFparm CPLpTable[] = { 
	IOPU("pos_nodes", CPL_POS_NODE, IF_VECTOR|IF_STRING, "in nodes"),
	IOPU("neg_nodes", CPL_NEG_NODE, IF_VECTOR|IF_STRING, "out nodes"),
	IOP("dimension", CPL_DIM, IF_INTEGER,               "number of coupled lines"),
	IOP("length",    CPL_LENGTH, IF_REAL,               "length of lines"),
};

IFparm CPLmPTable[] = { /* model parameters */
 IOP( "r",    CPL_R, IF_REALVEC,"resistance per length"),
 IOP( "l",    CPL_L, IF_REALVEC,"inductance per length"),
 IOP( "c",    CPL_C, IF_REALVEC,"capacitance per length"),
 IOP( "g",    CPL_G, IF_REALVEC,"conductance per length"),
 IOP( "length",    CPL_length, IF_REAL,"length"),
 IP( "cpl",  CPL_MOD_R,   IF_FLAG,"Device is a cpl model"),
};

char *CPLnames[] = {
	"P+",
	"P-"
};

int CPLnSize = NUMELEMS(CPLnames);
int CPLiSize = sizeof(CPLinstance);
int CPLmSize = sizeof(CPLmodel);
int CPLmPTSize = NUMELEMS(CPLmPTable);
int CPLpTSize = NUMELEMS(CPLpTable);
