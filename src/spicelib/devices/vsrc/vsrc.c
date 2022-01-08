/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "vsrcdefs.h"
#include "ngspice/suffix.h"

IFparm VSRCpTable[] = { /* parameters */
 IOPP("dc",      VSRC_DC,        IF_REAL   ,"DC value of source"),
 IOPPA("acmag",   VSRC_AC_MAG,    IF_REAL   ,"AC Magnitude"),
 IOPAAU("acphase", VSRC_AC_PHASE,  IF_REAL   ,"AC Phase"),
 /* Modified to allow print @vin[sin] A.Roldan */
 IOP ("pulse",   VSRC_PULSE,     IF_REALVEC,"Pulse description"),
 IOP ("sin",     VSRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IOPR("sine",    VSRC_SINE,      IF_REALVEC,"Sinusoidal source description"),
 IOP ("exp",     VSRC_EXP,       IF_REALVEC,"Exponential source description"),
 IOP ("pwl",     VSRC_PWL,       IF_REALVEC,"Piecewise linear description"),
 IOP ("sffm",    VSRC_SFFM,      IF_REALVEC,"Single freq. FM description"),
 IOP ("am",      VSRC_AM,        IF_REALVEC,"Amplitude modulation description"),
 IOP ("trnoise", VSRC_TRNOISE,   IF_REALVEC,"Transient noise description"),
 IOP ("trrandom", VSRC_TRRANDOM, IF_REALVEC,"random source description"),
#ifdef SHARED_MODULE
 IOP ("external", VSRC_EXTERNAL, IF_STRING,"external source description"),
#endif
  #ifdef RFSPICE
 IOP("portnum",   VSRC_PORTNUM,         IF_INTEGER,"Port index"),
 IOP("z0",        VSRC_PORTZ0,          IF_REAL,   "Port impedance"),
 IOP("pwr",       VSRC_PORTPWR,         IF_REAL,   "Port Power"),
 IOP("freq",      VSRC_PORTFREQ,        IF_REAL,   "Port frequency"),
 IOP("phase",    VSRC_PORTPHASE,       IF_REAL,   "Phase of the source"),
#endif
 OPU ("pos_node",VSRC_POS_NODE,  IF_INTEGER,"Positive node of source"),
 OPU ("neg_node",VSRC_NEG_NODE,  IF_INTEGER,"Negative node of source"),
 OPU ("function",VSRC_FCN_TYPE,  IF_INTEGER,"Function of the source"),
 OPU ("order",   VSRC_FCN_ORDER, IF_INTEGER,"Order of the source function"),
 OPU ("coeffs",  VSRC_FCN_COEFFS,IF_REALVEC,"Coefficients for the function"),
 OPU ("acreal",  VSRC_AC_REAL,   IF_REAL,   "AC real part"),
 OPU ("acimag",  VSRC_AC_IMAG,   IF_REAL,   "AC imaginary part"),
 IP  ("ac",      VSRC_AC,        IF_REALVEC,"AC magnitude, phase vector"),
 OP  ("i",       VSRC_CURRENT,   IF_REAL,   "Voltage source current"),
 OP  ("p",       VSRC_POWER,     IF_REAL,   "Instantaneous power"),
 IP  ("r",       VSRC_R,         IF_REAL,   "pwl repeat value"),
 IP  ("td",      VSRC_TD,        IF_REAL,   "pwl delay value"),
 IP  ("distof1", VSRC_D_F1,      IF_REALVEC,"f1 input for distortion"),
 IP  ("distof2", VSRC_D_F2,      IF_REALVEC,"f2 input for distortion")
};

char *VSRCnames[] = {
    "V+",
    "V-"
};

int     VSRCnSize = NUMELEMS(VSRCnames);
int     VSRCpTSize = NUMELEMS(VSRCpTable);
int     VSRCmPTSize = 0;
int     VSRCiSize = sizeof(VSRCinstance);
int     VSRCmSize = sizeof(VSRCmodel);
