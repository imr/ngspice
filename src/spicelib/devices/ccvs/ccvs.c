/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ccvsdefs.h"
#include "ngspice/suffix.h"

/* current controlled voltage source */
IFparm CCVSpTable[] = { /* parameters */ 
 IOPU("gain",     CCVS_TRANS,   IF_REAL    ,"Transresistance (gain)"),
 IOPU("control",  CCVS_CONTROL, IF_INSTANCE,"Controlling voltage source"),
 IP("sens_trans",CCVS_TRANS_SENS,IF_FLAG,  
        "flag to request sens. WRT transimpedance"),
 OPU("pos_node", CCVS_POS_NODE,IF_INTEGER, "Positive node of source"),
 OPU("neg_node", CCVS_NEG_NODE,IF_INTEGER, "Negative node of source"),
 OP("i",        CCVS_CURRENT, IF_REAL,    "CCVS output current"),
 OP("v",        CCVS_VOLTS, IF_REAL,    "CCVS output voltage"),
 OP("p",        CCVS_POWER,   IF_REAL,    "CCVS power"),
 OPU("sens_dc",   CCVS_QUEST_SENS_DC,  IF_REAL,"dc sensitivity "),
 OPU("sens_real", CCVS_QUEST_SENS_REAL,IF_REAL,"real part of ac sensitivity"),
 OPU("sens_imag", CCVS_QUEST_SENS_IMAG,IF_REAL,"imag part of ac sensitivity"),
 OPU("sens_mag",  CCVS_QUEST_SENS_MAG, IF_REAL,"sensitivity of ac magnitude"),
 OPU("sens_ph",   CCVS_QUEST_SENS_PH,  IF_REAL, "sensitivity of ac phase"),
 OPU("sens_cplx", CCVS_QUEST_SENS_CPLX,IF_COMPLEX,"ac sensitivity")
};

char *CCVSnames[] = {
    "H+",
    "H-"
};

int	CCVSnSize = NUMELEMS(CCVSnames);
int	CCVSpTSize = NUMELEMS(CCVSpTable);
int	CCVSmPTSize = 0;
int	CCVSiSize = sizeof(CCVSinstance);
int	CCVSmSize = sizeof(CCVSmodel);
