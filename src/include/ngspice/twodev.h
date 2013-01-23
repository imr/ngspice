/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * Two-Dimensional Numerical Device Data Structures 
 */
  
#ifndef ngspice_TWODEV_H
#define ngspice_TWODEV_H

#include "ngspice/gendev.h"
#include "ngspice/smpdefs.h"

typedef struct sTWOdevice
{
    double *dcSolution;                /* solution vector for device */
    double *dcDeltaSolution;           /* delta solution vector */
    double *copiedSolution;            /* copy of the solution vector */
    double *rhs;                       /* rhs vector */
    double *rhsImag;                   /* imaginary part of rhs vector */
    SMPmatrix *matrix;                 /* matrix for device equations */
    int solverType;                    /* what is matrix set up to do */
    int dimEquil;                      /* dimension in equilibrium */
    int numOrigEquil;                  /* orig number of nz's in equilibrium */
    int numFillEquil;                  /* fill number of nz's in equilibrium */
    int dimBias;                       /* dimension under bias */
    int numOrigBias;                   /* orig number of nz's under bias */
    int numFillBias;                   /* fill number of nz's under bias */
    int numEqns;                       /* number of equations */
    int poissonOnly;                   /* flag for Poisson eqn solution */
    struct sTWOelem **elements;        /* 1D list of elements */
    struct sTWOelem ***elemArray;      /* 2D list of elements for printing */
    double **devStates;                /* device states */
    double *xScale;                    /* array of X node locations */
    double *yScale;                    /* array of Y node locations */
    int numXNodes;                     /* number of X nodes */  
    int numYNodes;                     /* number of Y nodes */  
    int numNodes;                      /* total number of nodes */
    int numEdges;                      /* total number of edges */
    int numElems;                      /* total number of elements */
    struct sTWOcontact *pFirstContact; /* first contact */
    struct sTWOcontact *pLastContact;  /* last contact */
    struct sTWOchannel *pChannel;      /* surface mobility channel */
    struct sMaterialInfo *pMaterials;  /* temp-dep material information */
    struct sStatInfo *pStats;          /* run-time statistics */
    int converged;                     /* flag for device convergence */
    int iterationNumber;               /* device iteration counter */
    double width;                      /* device width in CM */
    double rhsNorm;                    /* norm of rhs vector */
    double abstol;                     /* absolute tolerance for device */
    double reltol;                     /* relative tolerance for device */
    char *name;	                       /* name of device */
} TWOdevice;

#define devState0 devStates[0]
#define devState1 devStates[1]
#define devState2 devStates[2]
#define devState3 devStates[3]
#define devState4 devStates[4]
#define devState5 devStates[5]
#define devState6 devStates[6]
#define devState7 devStates[7]

typedef struct sTWOcontact
{
    struct sTWOcontact *next;       /* pointer to next contact */
    struct sTWOnode **pNodes;	    /* pointer to the contact nodes */
    int numNodes;                   /* number of nodes in contact */
    int id;                         /* unique contact identifier */
    double workf;                   /* metal work function */
} TWOcontact;

/* Structure for channels.
 * A channel is divided into 'columns' that are perpendicular to the
 * channel's insulator/semiconductor interface.  Each column begins
 * at its 'seed', the closest semiconductor element to the interface.
 * It is assumed that the current flows parallel to the interface.
 */
typedef struct sTWOchannel
{
    struct sTWOchannel *next;      /* pointer to next channel */
    struct sTWOelem *pSeed;        /* pointer to the seed element */
    struct sTWOelem *pNElem;       /* pointer to the insulator element */
    int id;                        /* unique channel identifier */
    int type;                      /* ID of direction to interface */
} TWOchannel;

struct mosConductances
{
    double dIdDVdb;
    double dIdDVsb;
    double dIdDVgb;
    double dIsDVdb;
    double dIsDVsb;
    double dIsDVgb;
    double dIgDVdb;
    double dIgDVsb;
    double dIgDVgb;
};

#endif
