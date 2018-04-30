/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_METHDEFS_H
#define ngspice_METHDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sMETHcard {
    struct sMETHcard *METHnextCard;
    double METHdabstol;
    double METHdreltol;
    double METHomega;
    int METHoneCarrier;
    int METHacAnalysisMethod;
    int METHmobDeriv;
    int METHitLim;
    int METHvoltPred;
    unsigned int METHdabstolGiven : 1;
    unsigned int METHdreltolGiven : 1;
    unsigned int METHomegaGiven : 1;
    unsigned int METHoneCarrierGiven : 1;
    unsigned int METHacAnalysisMethodGiven : 1;
    unsigned int METHmobDerivGiven : 1;
    unsigned int METHitLimGiven : 1;
    unsigned int METHvoltPredGiven : 1;
} METHcard;

/* METH parameters */
enum {
    METH_DABSTOL = 1,
    METH_DRELTOL,
    METH_OMEGA,
    METH_ONEC,
    METH_ACANAL,
    METH_NOMOBDERIV,
    METH_ITLIM,
    METH_VOLTPRED,
};

#endif
