/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_DOMNDEFS_H
#define ngspice_DOMNDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sDOMNcard {
    struct sDOMNcard *DOMNnextCard;		/* pointer to next domain */
    double DOMNxLow;
    double DOMNxHigh;
    double DOMNyLow;
    double DOMNyHigh;
    int DOMNixLow;
    int DOMNixHigh;
    int DOMNiyLow;
    int DOMNiyHigh;
    int DOMNmaterial;
    int DOMNnumber;
    unsigned int DOMNxLowGiven : 1;
    unsigned int DOMNxHighGiven : 1;
    unsigned int DOMNyLowGiven : 1;
    unsigned int DOMNyHighGiven : 1;
    unsigned int DOMNixLowGiven : 1;
    unsigned int DOMNixHighGiven : 1;
    unsigned int DOMNiyLowGiven : 1;
    unsigned int DOMNiyHighGiven : 1;
    unsigned int DOMNmaterialGiven : 1;
    unsigned int DOMNnumberGiven : 1;
} DOMNcard;

/* DOMN parameters */
enum {
    DOMN_X_LOW = 1,
    DOMN_X_HIGH,
    DOMN_Y_LOW,
    DOMN_Y_HIGH,
    DOMN_IX_LOW,
    DOMN_IX_HIGH,
    DOMN_IY_LOW,
    DOMN_IY_HIGH,
    DOMN_NUMBER,
    DOMN_MATERIAL,
};

#endif
