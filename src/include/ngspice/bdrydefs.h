/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */


#ifndef ngspice_BDRYDEFS_H
#define ngspice_BDRYDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sBDRYcard {
    struct sBDRYcard *BDRYnextCard;
    double BDRYxLow;
    double BDRYxHigh;
    double BDRYyLow;
    double BDRYyHigh;
    double BDRYqf;
    double BDRYsn;
    double BDRYsp;
    double BDRYlayer;
    int BDRYixLow;
    int BDRYixHigh;
    int BDRYiyLow;
    int BDRYiyHigh;
    int BDRYdomain;
    int BDRYneighbor;
    unsigned int BDRYxLowGiven : 1;
    unsigned int BDRYxHighGiven : 1;
    unsigned int BDRYyLowGiven : 1;
    unsigned int BDRYyHighGiven : 1;
    unsigned int BDRYqfGiven : 1;
    unsigned int BDRYsnGiven : 1;
    unsigned int BDRYspGiven : 1;
    unsigned int BDRYlayerGiven : 1;
    unsigned int BDRYixLowGiven : 1;
    unsigned int BDRYixHighGiven : 1;
    unsigned int BDRYiyLowGiven : 1;
    unsigned int BDRYiyHighGiven : 1;
    unsigned int BDRYdomainGiven : 1;
    unsigned int BDRYneighborGiven : 1;
} BDRYcard;

/* BDRY parameters */
enum {
    BDRY_X_LOW = 1,
    BDRY_X_HIGH,
    BDRY_Y_LOW,
    BDRY_Y_HIGH,
    BDRY_IX_LOW,
    BDRY_IX_HIGH,
    BDRY_IY_LOW,
    BDRY_IY_HIGH,
    BDRY_DOMAIN,
    BDRY_NEIGHBOR,
    BDRY_QF,
    BDRY_SN,
    BDRY_SP,
    BDRY_LAYER,
};

#endif
