/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "cccsdefs.h"
#include "ngspice/suffix.h"

IFparm CCCSpTable[] = { /* parameters */ 
 IOPU("gain",    CCCS_GAIN,    IF_REAL    ,"Gain of source"),
 IOPU("control", CCCS_CONTROL, IF_INSTANCE,"Name of controlling source"),
 IOP ( "m",      CCCS_M,         IF_REAL   , "Parallel multiplier"),
 IP("sens_gain",CCCS_GAIN_SENS,IF_FLAG, "flag to request sensitivity WRT gain"),
 OPU("neg_node", CCCS_NEG_NODE,IF_INTEGER, "Negative node of source"),
 OPU("pos_node", CCCS_POS_NODE,IF_INTEGER, "Positive node of source"),
 OP("i",        CCCS_CURRENT, IF_REAL,    "CCCS output current"),
 OP("v",        CCCS_VOLTS,   IF_REAL,    "CCCS voltage at output"),
 OP("p",        CCCS_POWER,   IF_REAL,    "CCCS power"),
 OPU("sens_dc",  CCCS_QUEST_SENS_DC,  IF_REAL, "dc sensitivity "),
 OPU("sens_real",CCCS_QUEST_SENS_REAL,IF_REAL, "real part of ac sensitivity"),
 OPU("sens_imag",CCCS_QUEST_SENS_IMAG,IF_REAL, "imag part of ac sensitivity"),
 OPU("sens_mag", CCCS_QUEST_SENS_MAG, IF_REAL, "sensitivity of ac magnitude"),
 OPU("sens_ph",  CCCS_QUEST_SENS_PH,  IF_REAL, "sensitivity of ac phase"),
 OPU("sens_cplx",CCCS_QUEST_SENS_CPLX,IF_COMPLEX, "ac sensitivity")
};

#if 0
static IFparm CCCSmPTable[] = {
     /* model parameters */ 
};
#endif

char *CCCSnames[] = {
    "F+",
    "F-"
};

int	CCCSnSize = NUMELEMS(CCCSnames);
int	CCCSpTSize = NUMELEMS(CCCSpTable);
int	CCCSmPTSize = 0;
int	CCCSiSize = sizeof(CCCSinstance);
int	CCCSmSize = sizeof(CCCSmodel);
