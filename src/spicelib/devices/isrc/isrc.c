/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "isrcdefs.h"
#include "ngspice/suffix.h"

IFparm ISRCpTable[] = { /* parameters */
 IOPP("dc",      ISRC_DC,        IF_REAL   ,"DC value of source"),
 IOPPR("c",      ISRC_DC,        IF_REAL,   "Current through current source"),
 IOP ( "m",      ISRC_M,         IF_REAL   ,"Parallel multiplier"),
 IOPPA("acmag",   ISRC_AC_MAG,    IF_REAL   ,"AC Magnitude"),
 IOPAAU("acphase", ISRC_AC_PHASE,  IF_REAL   ,"AC Phase"),
 /* Modified to allow print @Iin[sin] A.Roldan */
 IOP ("pulse",   ISRC_PULSE,     IF_REALVEC,"Pulse description"),
 IOP ("sin",     ISRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IOPR("sine",    ISRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IOP ("exp",     ISRC_EXP,       IF_REALVEC,"Exponential source description"),
 IOP ("pwl",     ISRC_PWL,       IF_REALVEC,"Piecewise linear description"),
 IOP ("sffm",    ISRC_SFFM,      IF_REALVEC,"Single freq. FM description"),
 IOP ("am",      ISRC_AM,        IF_REALVEC,"Amplitude modulation description"),
 IOP ("trnoise", ISRC_TRNOISE,   IF_REALVEC,"Transient noise description"),
 IOP ("trrandom", ISRC_TRRANDOM, IF_REALVEC,"random source description"),
#ifdef SHARED_MODULE
 IOP ("external", ISRC_EXTERNAL, IF_STRING,"external source description"),
#endif
 OPU ("pos_node",ISRC_POS_NODE,  IF_INTEGER,"Positive node of source"),
 OPU ("neg_node",ISRC_NEG_NODE,  IF_INTEGER,"Negative node of source"),
 OPU ("function",ISRC_FCN_TYPE,  IF_INTEGER,"Function of the source"),
 OPU ("order",   ISRC_FCN_ORDER, IF_INTEGER,"Order of the source function"),
 OPU ("coeffs",  ISRC_FCN_COEFFS,IF_REALVEC,"Coefficients for the function"),
 OPU ("acreal",  ISRC_AC_REAL,   IF_REAL,   "AC real part"),
 OPU ("acimag",  ISRC_AC_IMAG,   IF_REAL,   "AC imaginary part"),
 IP  ("ac",      ISRC_AC,        IF_REALVEC,"AC magnitude, phase vector"),
 OP  ("v",       ISRC_VOLTS,     IF_REAL,   "Voltage across the supply"),
 OP  ("p",       ISRC_POWER,     IF_REAL,   "Power supplied by the source"),
 OP  ("current", ISRC_CURRENT,   IF_REAL,   "Current in DC or Transient mode"),
 IP  ("distof1", ISRC_D_F1,      IF_REALVEC,"f1 input for distortion"),
 IP  ("distof2", ISRC_D_F2,      IF_REALVEC,"f2 input for distortion")
};

char *ISRCnames[] = {
    "I+",
    "I-"
};

int     ISRCnSize = NUMELEMS(ISRCnames);
int     ISRCpTSize = NUMELEMS(ISRCpTable);
int     ISRCmPTSize = 0;
int     ISRCiSize = sizeof(ISRCinstance);
int     ISRCmSize = sizeof(ISRCmodel);
