/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_ELCTDEFS_H
#define ngspice_ELCTDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sELCTcard {
    struct sELCTcard *ELCTnextCard;
    double ELCTxLow;
    double ELCTxHigh;
    double ELCTyLow;
    double ELCTyHigh;
    int ELCTixLow;
    int ELCTixHigh;
    int ELCTiyLow;
    int ELCTiyHigh;
    int ELCTnumber;
    unsigned int ELCTxLowGiven : 1;
    unsigned int ELCTxHighGiven : 1;
    unsigned int ELCTyLowGiven : 1;
    unsigned int ELCTyHighGiven : 1;
    unsigned int ELCTixLowGiven : 1;
    unsigned int ELCTixHighGiven : 1;
    unsigned int ELCTiyLowGiven : 1;
    unsigned int ELCTiyHighGiven : 1;
    unsigned int ELCTnumberGiven : 1;
} ELCTcard;

/* ELCT parameters */
enum {
    ELCT_X_LOW = 1,
    ELCT_X_HIGH,
    ELCT_Y_LOW,
    ELCT_Y_HIGH,
    ELCT_IX_LOW,
    ELCT_IX_HIGH,
    ELCT_IY_LOW,
    ELCT_IY_HIGH,
    ELCT_NUMBER,
};

#endif
