/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "vccsdefs.h"
#include "ngspice/suffix.h"

IFparm VCCSpTable[] = { /* parameters */ 
 IOPU("gain",       VCCS_TRANS, IF_REAL, "Transconductance of source (gain)"),
 IOP ( "m",      VCCS_M,         IF_REAL   , "Parallel multiplier"),
 IP("sens_trans",  VCCS_TRANS_SENS,IF_FLAG,
        "flag to request sensitivity WRT transconductance"),
 OPU("pos_node",    VCCS_POS_NODE, IF_INTEGER, "Positive node of source"),
 OPU("neg_node",    VCCS_NEG_NODE, IF_INTEGER, "Negative node of source"),
 OPU("cont_p_node",VCCS_CONT_P_NODE,IF_INTEGER,
				"Positive node of contr. source"),
 OPU("cont_n_node",VCCS_CONT_N_NODE,IF_INTEGER,
				"Negative node of contr. source"),
 IP("ic",          VCCS_IC, IF_REAL, "Initial condition of controlling source"),
 OP("i",            VCCS_CURRENT,IF_REAL, "Output current"),
 OP("v",            VCCS_VOLTS,IF_REAL, "Voltage across output"),
 OP("p",            VCCS_POWER,  IF_REAL, "Power"),
 OPU("sens_dc",   VCCS_QUEST_SENS_DC,       IF_REAL,    "dc sensitivity "),
 OPU("sens_real", VCCS_QUEST_SENS_REAL, IF_REAL, "real part of ac sensitivity"),
 OPU("sens_imag", VCCS_QUEST_SENS_IMAG, IF_REAL, "imag part of ac sensitivity"),
 OPU("sens_mag",  VCCS_QUEST_SENS_MAG,  IF_REAL, "sensitivity of ac magnitude"),
 OPU("sens_ph",   VCCS_QUEST_SENS_PH,   IF_REAL,  "sensitivity of ac phase"),
 OPU("sens_cplx", VCCS_QUEST_SENS_CPLX, IF_COMPLEX,    "ac sensitivity")
};

char *VCCSnames[] = {
    "V+",
    "V-",
    "VC+",
    "VC-"
};

int	VCCSnSize = NUMELEMS(VCCSnames);
int	VCCSpTSize = NUMELEMS(VCCSpTable);
int	VCCSmPTSize = 0;
int	VCCSiSize = sizeof(VCCSinstance);
int	VCCSmSize = sizeof(VCCSmodel);
