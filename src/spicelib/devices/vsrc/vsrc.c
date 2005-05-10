/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "devdefs.h"
#include "ifsim.h"
#include "vsrcdefs.h"
#include "suffix.h"

IFparm VSRCpTable[] = { /* parameters */ 
 IOPP("dc",      VSRC_DC,        IF_REAL   ,"D.C. source value"),
 IOPPA("acmag",   VSRC_AC_MAG,    IF_REAL   ,"A.C. Magnitude"),
 IOPAAU("acphase", VSRC_AC_PHASE,  IF_REAL   ,"A.C. Phase"),
 IP ("pulse",   VSRC_PULSE,     IF_REALVEC,"Pulse description"),
 IP ("sine",    VSRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IP ("sin",     VSRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IP ("exp",     VSRC_EXP,       IF_REALVEC,"Exponential source description"),
 IP ("pwl",     VSRC_PWL,       IF_REALVEC,"Piecewise linear description"),
 IP ("sffm",    VSRC_SFFM,      IF_REALVEC,"Single freq. FM descripton"),
 IP ("am",      VSRC_AM,        IF_REALVEC,"Amplitude modulation descripton"),
 OPU ("pos_node",VSRC_POS_NODE,  IF_INTEGER,"Positive node of source"),
 OPU ("neg_node",VSRC_NEG_NODE,  IF_INTEGER,"Negative node of source"),
 OPU ("function",VSRC_FCN_TYPE,  IF_INTEGER,"Function of the source"),
 OPU ("order",   VSRC_FCN_ORDER, IF_INTEGER,"Order of the source function"),
 OPU ("coeffs",  VSRC_FCN_COEFFS,IF_REALVEC,"Coefficients for the function"),
 OPU ("acreal",  VSRC_AC_REAL,   IF_REAL,   "AC real part"),
 OPU ("acimag",  VSRC_AC_IMAG,   IF_REAL,   "AC imaginary part"),
 IP ("ac",      VSRC_AC,        IF_REALVEC,"AC magnitude, phase vector"),
 OP ("i",       VSRC_CURRENT,   IF_REAL,   "Voltage source current"),
 OP ("p",       VSRC_POWER,     IF_REAL,   "Instantaneous power"),
 IP ("distof1", VSRC_D_F1,      IF_REALVEC,"f1 input for distortion"),
 IP ("distof2", VSRC_D_F2,      IF_REALVEC,"f2 input for distortion")
};

char *VSRCnames[] = {
    "V+",
    "V-"
};

int	VSRCnSize = NUMELEMS(VSRCnames);
int	VSRCpTSize = NUMELEMS(VSRCpTable);
int	VSRCmPTSize = 0;
int	VSRCiSize = sizeof(VSRCinstance);
int	VSRCmSize = sizeof(VSRCmodel);
