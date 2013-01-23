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
#define OPTN_RESISTOR	1
#define OPTN_CAPACITOR   2
#define OPTN_DIODE       3
#define OPTN_BIPOLAR     4
#define OPTN_SOIBJT      5
#define OPTN_MOSCAP      6
#define OPTN_MOSFET      7
#define OPTN_SOIMOS      8
#define OPTN_JFET        9
#define OPTN_MESFET      10
#define OPTN_DEFA        11
#define OPTN_DEFW        12
#define OPTN_DEFL        13
#define OPTN_BASE_AREA   14
#define OPTN_BASE_LENGTH 15
#define OPTN_BASE_DEPTH  16
#define OPTN_TNOM        17
#define OPTN_IC_FILE     18
#define OPTN_UNIQUE      19

#endif
