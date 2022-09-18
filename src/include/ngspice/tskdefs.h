/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */
#ifndef ngspice_TSKDEFS_H
#define ngspice_TSKDEFS_H


#include "ngspice/typedefs.h"
#include "ngspice/jobdefs.h"

struct TSKtask {
    JOB taskOptions;    /* job structure at the front to hold options */
    JOB *jobs;
    char *TSKname;
    double TSKtemp;
    double TSKnomTemp;
    int TSKmaxOrder;        /* maximum integration method order */
    int TSKintegrateMethod; /* the integration method to be used */
    double TSKxmu;          /* for trapezoidal method */
    int TSKindverbosity;    /* control check of inductive systems */
    int TSKcurrentAnalysis; /* the analysis in progress (if any) */

/* defines for the value of  TSKcurrentAnalysis */
#define DOING_DCOP 1
#define DOING_TRCV 2
#define DOING_AC   4
#define DOING_TRAN 8
#ifdef RFSPICE
#define DOING_SP   16
#endif

    int TSKbypass;
    int TSKdcMaxIter;       /* iteration limit for dc op.  (itl1) */
    int TSKdcTrcvMaxIter;   /* iteration limit for dc tran. curv (itl2) */
    int TSKtranMaxIter;     /* iteration limit for each timepoint for tran*/
                            /* (itl4) */
    int TSKnumSrcSteps;     /* number of steps for source stepping */
    int TSKnumGminSteps;    /* number of steps for Gmin stepping */
    double TSKgminFactor;   /* factor for Gmin stepping */
    double TSKminBreak;     /* minimum time difference between breakpoints */
    double TSKabstol;
    double TSKpivotAbsTol;
    double TSKpivotRelTol;
    double TSKreltol;
    double TSKchgtol;
    double TSKvoltTol;
#ifdef NEWTRUNC
    double TSKlteReltol;
    double TSKlteAbstol;
#endif /* NEWTRUNC */
    double TSKgmin;
    double TSKgshunt;   /* shunt conductance (CKTdiagGmin) */
    double TSKcshunt;   /* shunt capacitor to ground */
    double TSKdelmin;
    double TSKtrtol;
    double TSKdefaultMosM;
    double TSKdefaultMosL;
    double TSKdefaultMosW;
    double TSKdefaultMosAD;
    double TSKdefaultMosAS;
    unsigned int TSKfixLimit:1;
    unsigned int TSKnoOpIter:1; /* no OP iterating, go straight to gmin step */
    unsigned int TSKtryToCompact:1; /* flag for LTRA lines */
    unsigned int TSKbadMos3:1; /* flag for MOS3 models */
    unsigned int TSKkeepOpInfo:1; /* flag for small signal analyses */
    unsigned int TSKcopyNodesets:1; /* flag for nodeset copy */
    unsigned int TSKnodeDamping:1;  /* flag for node damping */
    unsigned int TSKnoopac:1; /* flag for no OP calculation before AC */
    double TSKabsDv;                 /* abs limit for iter-iter voltage change */
    double TSKrelDv;                 /* rel limit for iter-iter voltage change */
    double TSKepsmin;         /* minimum value for log */
};

#endif
