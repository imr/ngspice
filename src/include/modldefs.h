/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef MODL_H
#define MODL_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sMODLcard {
    struct sMODLcard *MODLnextCard;
    int MODLbandGapNarrowing;
    int MODLtempDepMobility;
    int MODLconcDepMobility;
    int MODLfieldDepMobility;
    int MODLtransDepMobility;
    int MODLsurfaceMobility;
    int MODLmatchingMobility;
    int MODLsrh;
    int MODLconcDepLifetime;
    int MODLauger;
    int MODLavalancheGen;
    unsigned int MODLbandGapNarrowingGiven : 1;
    unsigned int MODLtempDepMobilityGiven : 1;
    unsigned int MODLconcDepMobilityGiven : 1;
    unsigned int MODLfieldDepMobilityGiven : 1;
    unsigned int MODLtransDepMobilityGiven : 1;
    unsigned int MODLsurfaceMobilityGiven : 1;
    unsigned int MODLmatchingMobilityGiven : 1;
    unsigned int MODLsrhGiven : 1;
    unsigned int MODLconcDepLifetimeGiven : 1;
    unsigned int MODLaugerGiven : 1;
    unsigned int MODLavalancheGenGiven : 1;
} MODLcard;

/* MODL parameters */
#define MODL_BGNW	1
#define MODL_TEMPMOB	2
#define MODL_CONCMOB	3
#define MODL_FIELDMOB	4
#define MODL_TRANSMOB	5
#define MODL_SURFMOB	6
#define MODL_MATCHMOB	7
#define MODL_SRH	8
#define MODL_CONCTAU	9
#define MODL_AUGER	10
#define MODL_AVAL	11

#endif /* MODL_H */
