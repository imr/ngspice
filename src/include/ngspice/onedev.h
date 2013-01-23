/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * One Dimensional Numerical Device Data Structures
 */
 
#ifndef ngspice_ONEDEV_H
#define ngspice_ONEDEV_H

#include "ngspice/gendev.h"
#include "ngspice/smpdefs.h"

typedef struct sONEdevice
{
    double *dcSolution;                /* solution vector for device */
    double *dcDeltaSolution;           /* delta solution vector */
    double *copiedSolution;            /* copy of the solution vector */
    double *rhs;                       /* rhs vector */
    double *rhsImag;                   /* imaginary part of rhs vector */
    SMPmatrix *matrix;                 /* matrix for device equations */
    int solverType;                    /* type of equations matrix can solve */
    int dimEquil;                      /* dimension in equilibrium */
    int numOrigEquil;                  /* orig number of nz's in equilibrium */
    int numFillEquil;                  /* fill number of nz's in equilibrium */
    int dimBias;                       /* dimension under bias */
    int numOrigBias;                   /* orig number of nz's under bias */
    int numFillBias;                   /* fill number of nz's under bias */
    int numEqns;                       /* number of equations */
    int poissonOnly;                   /* flag for Poisson eqn solution */
    struct sONEelem **elemArray;       /* array of elements */
    double **devStates;                /* device states */
    int numNodes;                      /* total number of nodes */
    struct sONEcontact *pFirstContact; /* first contact */
    struct sONEcontact *pLastContact;  /* last contact */
    struct sMaterialInfo *pMaterials;  /* temp-dep material information */
    struct sStatInfo *pStats;          /* run-time statistics */
    int converged;                     /* flag for device convergence */
    int iterationNumber;               /* device iteration counter */
    int baseIndex;                     /* index for base contact in BJTs */
    double baseLength;		         /* length of base contact in BJTs */
    double area;                       /* area of device in CM^2 */
    double rhsNorm;                    /* norm of rhs vector */
    double abstol;                     /* absolute tolerance for device */
    double reltol;                     /* relative tolerance for device */
    char *name;                        /* name of device */
} ONEdevice;

#define devState0 devStates[0]
#define devState1 devStates[1]
#define devState2 devStates[2]
#define devState3 devStates[3]
#define devState4 devStates[4]
#define devState5 devStates[5]
#define devState6 devStates[6]
#define devState7 devStates[7]

typedef struct sONEcontact
{
    struct sONEcontact *next;       /* pointer to next contact */
    struct sONEnode **pNodes;	   /* pointer to the contact nodes */
    int numNodes;                   /* number of nodes in contact */
    int id;                         /* unique contact identifier */
    double workf;                   /* metal work function */
} ONEcontact;

#endif
