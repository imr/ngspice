/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef DOP_H
#define DOP_H

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
#define DOP_UNIF	1
#define DOP_LINEAR	2
#define DOP_GAUSS	3
#define DOP_ERFC	4
#define DOP_EXP		5
#define DOP_SUPREM3	6
#define DOP_ASCII	7
#define DOP_SUPASCII	8
#define DOP_INFILE	9
#define DOP_BORON	10
#define DOP_PHOSP	11
#define DOP_ARSEN	12
#define DOP_ANTIM	13
#define DOP_P_TYPE	14
#define DOP_N_TYPE	15
#define DOP_X_AXIS	16
#define DOP_Y_AXIS	17
#define DOP_X_LOW	18
#define DOP_X_HIGH	19
#define DOP_Y_LOW	20
#define DOP_Y_HIGH	21
#define DOP_CONC	22
#define DOP_LOCATION	23
#define DOP_CHAR_LEN	24
#define DOP_RATIO_LAT	25
#define DOP_ROTATE_LAT	26
#define DOP_UNIF_LAT	27
#define DOP_LINEAR_LAT	28
#define DOP_GAUSS_LAT	29
#define DOP_ERFC_LAT	30
#define DOP_EXP_LAT	31
#define DOP_DOMAIN	32

#endif /* DOP_H */
