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
#define BDRY_X_LOW	1
#define BDRY_X_HIGH	2
#define BDRY_Y_LOW	3
#define BDRY_Y_HIGH	4
#define BDRY_IX_LOW	5
#define BDRY_IX_HIGH	6
#define BDRY_IY_LOW	7
#define BDRY_IY_HIGH	8
#define BDRY_DOMAIN	9
#define BDRY_NEIGHBOR	10
#define BDRY_QF		11
#define BDRY_SN		12
#define BDRY_SP		13
#define BDRY_LAYER	14

#endif
