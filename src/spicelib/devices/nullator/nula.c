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
#include "nuladefs.h"
#include "ngspice/suffix.h"

IFparm NULApTable[] = { /* parameters */ 
 IOP("offset", NULA_OFFSET, IF_REAL,"Control offset"),
 OPU("cont_p_node",NULA_CONT_P_NODE,IF_INTEGER,
					"Positive node of nullator"),
 OPU("cont_n_node",NULA_CONT_N_NODE,IF_INTEGER,
					"Negative node of nullator"),
};

char *NULAnames[] = {
    "VC+",
    "VC-"
};

int	NULAnSize = NUMELEMS(NULAnames);
int	NULApTSize = NUMELEMS(NULApTable);
int	NULAmPTSize = 0;
int	NULAiSize = sizeof(NULAinstance);
int	NULAmSize = sizeof(NULAmodel);
