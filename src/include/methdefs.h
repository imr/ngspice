/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef METH_H
#define METH_H

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
#define METH_DABSTOL	1
#define METH_DRELTOL	2
#define METH_OMEGA	3
#define METH_ONEC	4
#define METH_ACANAL	5
#define METH_NOMOBDERIV	6
#define METH_ITLIM      7
#define METH_VOLTPRED	8

#endif /* METH_H */
