/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * One-Dimensional Element-based Simulation-Mesh Data Structures 
 */
 
#ifndef ngspice_ONEMESH_H
#define ngspice_ONEMESH_H

#include "ngspice/material.h"

typedef struct sONEelem {
    struct sONEelem *pElems[2];     /* array to store neighbor elements */
    struct sONEnode *pNodes[2];     /* array to store the element nodes */
    struct sONEedge *pEdge;         /* the edge */
    double dx;                      /* the x length */
    double rDx;                     /* 1 / dx */
    int domain;                     /* domain to which this elem belongs */
    int elemType;                   /* material type of element */
    ONEmaterial *matlInfo;          /* material information */
    double epsRel;                  /* epsilon of material rel to Semicon */
    int evalNodes[2];               /* nodes to be evaluated in elem */
} ONEelem;

#define pLeftElem pElems[0]
#define pRightElem pElems[1]

#define pLeftNode pNodes[0]
#define pRightNode pNodes[1]

typedef struct sONEedge {
    double mun;                      /* electron mobility */
    double mup;                      /* hole mobility */
    double dPsi;                     /* deltaPsi */
    double jn;                       /* electron current */
    double jp;                       /* hole current */
    double jd;                       /* displacement current */
    double dJnDpsiP1;                /* dJn/dPsi(+1) */
    double dJnDn;                    /* dJnx/dN */
    double dJnDnP1;                  /* dJn/dN(+1) */
    double dJpDpsiP1;                /* dJpx/dPsi(+1) */
    double dJpDp;                    /* dJpx/dP */
    double dJpDpP1;                  /* dJpxDp(+1) */
    double dCBand;                   /* effective delta conduction band */
    double dVBand;                   /* effective delta valence band */
    int edgeState;                   /* pointer to state vector */
    unsigned evaluated : 1;          /* flag to indicated evaluated */
} ONEedge;
	
typedef struct sONEnode {
    double x;                        /* x location */
    int nodeI;                       /* node x-index */
    int poiEqn;                      /* equation number for equil poisson */
    int psiEqn;                      /* equation number for bias poisson */
    int nEqn;                        /* equation number for n continuity */
    int pEqn;                        /* equation number for p continuity */
    int nodeType;                    /* type of node */
    int baseType;                    /* baseType n or p or none */
    double vbe;                      /* base emitter voltage for BJTs */
    struct sONEelem *pElems[2];      /* array of elements */
    double psi0;                     /* equilibrium potential */
    double psi;                      /* electrostatic potential */
    double nConc;                    /* electron conc. */
    double pConc;                    /* hole conc. */
    double nie;                      /* effective intrinsic carrier conc. */
    double eg;                       /* energy gap */
    double eaff;                     /* electron affinity; work phi for metal*/
    double tn;                       /* electron lifetime */
    double tp;                       /* hole lifetime */
    double netConc;                  /* net conc. */
    double totalConc;                /* total conc. */
    double na;                       /* acceptor conc. */
    double nd;                       /* donor conc. */
    double qf;                       /* fixed charge density */
    double nPred;                    /* predicted electron conc. */
    double pPred;                    /* predicted hole conc. */
    double uNet;                     /* net recombination rate */
    double dUdN;                     /* dU / dN */  
    double dUdP;                     /* dU / dP */  
    double dNdT;                     /* dN / dT value */
    double dPdT;                     /* dN / dT value */
    int nodeState;                   /* pointer to the state vector */
    unsigned evaluated : 1;          /* flag to indicated evaluated */
    /* sparse matrix pointers. pointers to doubles */
    double *fPsiPsiiM1;
    double *fPsiPsi;
    double *fPsiPsiiP1;
    double *fPsiN;
    double *fPsiP;
    double *fNPsiiM1;
    double *fNPsi;
    double *fNPsiiP1;
    double *fNNiM1;
    double *fNN;
    double *fNNiP1;
    double *fNPiM1;
    double *fNP;
    double *fNPiP1;
    double *fPPsiiM1;
    double *fPPsi;
    double *fPPsiiP1;
    double *fPPiM1;
    double *fPP;
    double *fPPiP1;
    double *fPNiM1;
    double *fPN;
    double *fPNiP1;
} ONEnode;


#define pLeftElem pElems[0]
#define pRightElem pElems[1]

#define nodePsi   nodeState 
#define nodeN     nodeState+1
#define nodeP     nodeState+3

#define edgeDpsi  edgeState

#define ONEnumNodeStates 5
#define ONEnumEdgeStates 2

#endif
