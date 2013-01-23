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
#define DOMN_X_LOW	1
#define DOMN_X_HIGH	2
#define DOMN_Y_LOW	3
#define DOMN_Y_HIGH	4
#define DOMN_IX_LOW	5
#define DOMN_IX_HIGH	6
#define DOMN_IY_LOW	7
#define DOMN_IY_HIGH	8
#define DOMN_NUMBER	9
#define DOMN_MATERIAL	10


#endif
