/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_MOBDEFS_H
#define ngspice_MOBDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sMOBcard {
    struct sMOBcard *MOBnextCard;
    int MOBmaterial;
    int MOBcarrier;
    int MOBcarrType;
    double MOBmuMax;
    double MOBmuMin;
    double MOBntRef;
    double MOBntExp;
    double MOBvSat;
    double MOBvWarm;
    double MOBmus;
    double MOBecA;
    double MOBecB;
    int MOBconcModel;
    int MOBfieldModel;
    int MOBinit;
    unsigned MOBmaterialGiven : 1;
    unsigned MOBcarrierGiven : 1;
    unsigned MOBcarrTypeGiven : 1;
    unsigned MOBmuMaxGiven : 1;
    unsigned MOBmuMinGiven : 1;
    unsigned MOBntRefGiven : 1;
    unsigned MOBntExpGiven : 1;
    unsigned MOBvSatGiven : 1;
    unsigned MOBvWarmGiven : 1;
    unsigned MOBmusGiven : 1;
    unsigned MOBecAGiven : 1;
    unsigned MOBecBGiven : 1;
    unsigned MOBconcModelGiven : 1;
    unsigned MOBfieldModelGiven : 1;
    unsigned MOBinitGiven : 1;
} MOBcard;

/* MOB parameters */
enum {
    MOB_ELEC = 1,
    MOB_HOLE,
    MOB_MAJOR,
    MOB_MINOR,
    MOB_MUMAX,
    MOB_MUMIN,
    MOB_NTREF,
    MOB_NTEXP,
    MOB_VSAT,
    MOB_VWARM,
    MOB_MUS,
    MOB_EC_A,
    MOB_EC_B,
    MOB_CONC_MOD,
    MOB_FIELD_MOD,
    MOB_MATERIAL,
    MOB_INIT,
};

#endif
