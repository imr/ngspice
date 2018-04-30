/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_DOPDEFS_H
#define ngspice_DOPDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sDOPcard {
    struct sDOPcard *DOPnextCard;
    double DOPxLow;
    double DOPxHigh;
    double DOPyLow;
    double DOPyHigh;
    double DOPconc;
    double DOPlocation;
    double DOPcharLen;
    double DOPratioLat;
    int DOPprofileType;
    int DOPlatProfileType;
    int DOProtateLat;
    int DOPimpurityType;
    int DOPaxisType;
    int DOPnumDomains;
    int *DOPdomains;
    char *DOPinFile;
    unsigned int DOPxLowGiven : 1;
    unsigned int DOPxHighGiven : 1;
    unsigned int DOPyLowGiven : 1;
    unsigned int DOPyHighGiven : 1;
    unsigned int DOPconcGiven : 1;
    unsigned int DOPlocationGiven : 1;
    unsigned int DOPcharLenGiven : 1;
    unsigned int DOPratioLatGiven : 1;
    unsigned int DOPprofileTypeGiven : 1;
    unsigned int DOPlatProfileTypeGiven : 1;
    unsigned int DOProtateLatGiven : 1;
    unsigned int DOPimpurityTypeGiven : 1;
    unsigned int DOPaxisTypeGiven : 1;
    unsigned int DOPdomainsGiven : 1;
    unsigned int DOPinFileGiven : 1;
} DOPcard;

/* DOP parameters */
enum {
    DOP_UNIF = 1,
    DOP_LINEAR,
    DOP_GAUSS,
    DOP_ERFC,
    DOP_EXP,
    DOP_SUPREM3,
    DOP_ASCII,
    DOP_SUPASCII,
    DOP_INFILE,
    DOP_BORON,
    DOP_PHOSP,
    DOP_ARSEN,
    DOP_ANTIM,
    DOP_P_TYPE,
    DOP_N_TYPE,
    DOP_X_AXIS,
    DOP_Y_AXIS,
    DOP_X_LOW,
    DOP_X_HIGH,
    DOP_Y_LOW,
    DOP_Y_HIGH,
    DOP_CONC,
    DOP_LOCATION,
    DOP_CHAR_LEN,
    DOP_RATIO_LAT,
    DOP_ROTATE_LAT,
    DOP_UNIF_LAT,
    DOP_LINEAR_LAT,
    DOP_GAUSS_LAT,
    DOP_ERFC_LAT,
    DOP_EXP_LAT,
    DOP_DOMAIN,
};

#endif
