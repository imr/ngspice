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
#define CONT_NEUTRAL	1
#define CONT_ALUMINUM	2
#define CONT_P_POLY	3
#define CONT_N_POLY	4
#define CONT_WORKFUN	5
#define CONT_NUMBER	6

#endif
