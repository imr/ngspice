/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_MODLDEFS_H
#define ngspice_MODLDEFS_H

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
enum {
    MODL_BGNW = 1,
    MODL_TEMPMOB,
    MODL_CONCMOB,
    MODL_FIELDMOB,
    MODL_TRANSMOB,
    MODL_SURFMOB,
    MODL_MATCHMOB,
    MODL_SRH,
    MODL_CONCTAU,
    MODL_AUGER,
    MODL_AVAL,
};

#endif
