/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_CONTDEFS_H
#define ngspice_CONTDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sCONTcard {
    struct sCONTcard *CONTnextCard;
    double CONTworkfun;
    int CONTtype;
    int CONTnumber;
    unsigned int CONTworkfunGiven : 1;
    unsigned int CONTtypeGiven : 1;
    unsigned int CONTnumberGiven : 1;
} CONTcard;

/* CONT parameters */
enum {
    CONT_NEUTRAL = 1,
    CONT_ALUMINUM,
    CONT_P_POLY,
    CONT_N_POLY,
    CONT_WORKFUN,
    CONT_NUMBER,
};

#endif
