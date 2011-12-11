/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "vcvsdefs.h"
#include "ngspice/suffix.h"

IFparm VCVSpTable[] = { /* parameters */ 
 IOPU("gain", VCVS_GAIN, IF_REAL,"Voltage gain"),
 IP("sens_gain",VCVS_GAIN_SENS,IF_FLAG,"flag to request sensitivity WRT gain"),
 OPU("pos_node", VCVS_POS_NODE, IF_INTEGER, "Positive node of source"),
 OPU("neg_node", VCVS_NEG_NODE, IF_INTEGER, "Negative node of source"),
 OPU("cont_p_node",VCVS_CONT_P_NODE,IF_INTEGER,
					"Positive node of contr. source"),
 OPU("cont_n_node",VCVS_CONT_N_NODE,IF_INTEGER,
					"Negative node of contr. source"),
 IP("ic", VCVS_IC, IF_REAL, "Initial condition of controlling source"),
 OP("i",        VCVS_CURRENT,       IF_REAL,        "Output current"),
 OP("v",        VCVS_VOLTS,         IF_REAL,        "Output voltage"),
 OP("p",        VCVS_POWER,         IF_REAL,        "Power"),
 OPU("sens_dc",   VCVS_QUEST_SENS_DC,  IF_REAL, "dc sensitivity "),
 OPU("sens_real", VCVS_QUEST_SENS_REAL,IF_REAL, "real part of ac sensitivity"),
 OPU("sens_imag", VCVS_QUEST_SENS_IMAG,IF_REAL, "imag part of ac sensitivity"),
 OPU("sens_mag",  VCVS_QUEST_SENS_MAG, IF_REAL, "sensitivity of ac magnitude"),
 OPU("sens_ph",   VCVS_QUEST_SENS_PH,  IF_REAL, "sensitivity of ac phase"),
 OPU("sens_cplx", VCVS_QUEST_SENS_CPLX,     IF_COMPLEX,    "ac sensitivity")
};

char *VCVSnames[] = {
    "V+",
    "V-",
    "VC+",
    "VC-"
};

int	VCVSnSize = NUMELEMS(VCVSnames);
int	VCVSpTSize = NUMELEMS(VCVSpTable);
int	VCVSmPTSize = 0;
int	VCVSiSize = sizeof(VCVSinstance);
int	VCVSmSize = sizeof(VCVSmodel);
