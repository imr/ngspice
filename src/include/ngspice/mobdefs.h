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
#define MOB_ELEC	1
#define MOB_HOLE	2
#define MOB_MAJOR	3
#define MOB_MINOR	4
#define MOB_MUMAX	5
#define MOB_MUMIN	6
#define MOB_NTREF	7
#define MOB_NTEXP	8
#define MOB_VSAT	9
#define MOB_VWARM	10
#define MOB_MUS		11
#define MOB_EC_A	12
#define MOB_EC_B	13
#define MOB_CONC_MOD	14
#define MOB_FIELD_MOD	15
#define MOB_MATERIAL	16
#define MOB_INIT	17

#endif
