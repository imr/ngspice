/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors:  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_OPTNDEFS_H
#define ngspice_OPTNDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sOPTNcard {
    struct sOPTNcard *OPTNnextCard;
    char *OPTNicFile;
    int OPTNunique;
    int OPTNdeviceType;
    double OPTNdefa;
    double OPTNdefw;
    double OPTNdefl;
    double OPTNbaseArea;
    double OPTNbaseLength;
    double OPTNbaseDepth;
    double OPTNtnom;
    unsigned int OPTNicFileGiven : 1;
    unsigned int OPTNuniqueGiven : 1;
    unsigned int OPTNdeviceTypeGiven : 1;
    unsigned int OPTNdefaGiven : 1;
    unsigned int OPTNdefwGiven : 1;
    unsigned int OPTNdeflGiven : 1;
    unsigned int OPTNbaseAreaGiven : 1;
    unsigned int OPTNbaseLengthGiven : 1;
    unsigned int OPTNbaseDepthGiven : 1;
    unsigned int OPTNtnomGiven : 1;
} OPTNcard;

/* OPTN parameters */
enum {
    OPTN_RESISTOR = 1,
    OPTN_CAPACITOR,
    OPTN_DIODE,
    OPTN_BIPOLAR,
    OPTN_SOIBJT,
    OPTN_MOSCAP,
    OPTN_MOSFET,
    OPTN_SOIMOS,
    OPTN_JFET,
    OPTN_MESFET,
    OPTN_DEFA,
    OPTN_DEFW,
    OPTN_DEFL,
    OPTN_BASE_AREA,
    OPTN_BASE_LENGTH,
    OPTN_BASE_DEPTH,
    OPTN_TNOM,
    OPTN_IC_FILE,
    OPTN_UNIQUE,
};

#endif
