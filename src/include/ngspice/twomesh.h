/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * Two-Dimensional Element-based Simulation-Mesh Data Structures 
 */
 
#ifndef ngspice_TWOMESH_H
#define ngspice_TWOMESH_H

#include "ngspice/material.h"

typedef struct sTWOelem
{
    struct sTWOelem *pElems[4];     /* array to store the element neighbors */
    struct sTWOnode *pNodes[4];     /* array to store the element nodes */
    struct sTWOedge *pEdges[4];     /* array to store the edges */
    double dx;                      /* the x length */
    double dy;                      /* the y length */
    double dxOverDy;                /* dX / dY */
    double dyOverDx;                /* dY / dX */
    int domain;                     /* device domain owning element */
    int elemType;                   /* material type of element */
    TWOmaterial *matlInfo;          /* material information */
    double epsRel;                  /* relative epsilon */
/* Values needed to calc. mobility and its derivatives at element center */
    double mun0, mup0;              /* temp. and doping-dep. mobilities */
    double mun, mup;                /* field and carrier-dep. mobilities */
    double dMunDEs, dMupDEs;        /* Mob Deriv surf EFIELD Comp */
    double dMunDEx, dMupDEx;        /* Mob Deriv x EFIELD Comp */
    double dMunDEy, dMupDEy;        /* Mob Deriv y EFIELD Comp */
    double dMunDWx, dMupDWx;        /* Mob Deriv x WDF Comp */
    double dMunDWy, dMupDWy;        /* Mob Deriv y WDF Comp */
    double dMunDN,  dMupDN;	    /* Mob Deriv nConc Comp */
    double dMunDP,  dMupDP;	    /* Mob Deriv pConc Comp */
    unsigned surface : 1;           /* flag to indicate surface elem */
    int channel;	            /* id of channel elem is in, 0 is none */
    int direction;		    /* direction of flow for channels */
    int evalNodes[4];               /* nodes to be evaluated in elem */
    int evalEdges[4];               /* edges to be evaluated in elem */
} TWOelem;

#define pTopElem   pElems[0]
#define pRightElem pElems[1]
#define pBotElem   pElems[2]
#define pLeftElem  pElems[3]

#define pTLNode pNodes[0]
#define pTRNode pNodes[1]
#define pBRNode pNodes[2]
#define pBLNode pNodes[3]

#define pTopEdge   pEdges[0]
#define pRightEdge pEdges[1]
#define pBotEdge   pEdges[2]
#define pLeftEdge  pEdges[3]

typedef struct sTWOedge
{
    int edgeType;                    /* boundary type of edge */
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
    double qf;                       /* fixed charge density */
  /* Terms to find weighted carrier driving force and its derivatives */
    double wdfn;                     /* N weighted driving force */
    double wdfp;                     /* P weighted driving force */
    double dWnDpsiP1;                /* dWn/dPsi(+1) */
    double dWnDn;                    /* dWn/dN */
    double dWnDnP1;                  /* dWn/dN(+1) */
    double dWpDpsiP1;                /* dWp/dPsi(+1) */
    double dWpDp;                    /* dWp/dP */
    double dWpDpP1;                  /* dWp/dP(+1) */
  /* Coefficients for weighting mobility on sides of edges */
    double kNeg;		     /* Spline for negative side of edge */
    double kPos;		     /* Spline for positive side of edge */

    int edgeState;                   /* pointer to state vector */
    unsigned evaluated : 1;          /* flag to indicated evaluated */
} TWOedge;

typedef struct sTWOnode {
    int nodeType;                    /* type of node */
    int nodeI;                       /* node x-index */
    int nodeJ;                       /* node y-index */
    int poiEqn;			     /* equation number for equilib poisson */
    int psiEqn;			     /* equation number for bias poisson */
    int nEqn;			     /* equation number for n continuity */
    int pEqn;			     /* equation number for p continuity */
    struct sTWOelem *pElems[4];      /* array of elements */
    double psi0;                     /* equilibrium potential */
    double psi;                      /* electrostatic potential */
    double nConc;                    /* electron conc. */
    double pConc;                    /* hole conc. */
    double nie;                      /* effective intrinsic carrier conc. */
    double eg;                       /* energy gap */
    double eaff;                     /* electron affinity; work phi for metal*/
    double tn;                       /* electron lifetime */
    double tp;                       /* hole lifetime */
    double netConc;                  /* net doping conc. */
    double totalConc;                /* total doping conc. */
    double na;                       /* acceptor conc. */
    double nd;                       /* donor conc. */
    double nPred;                    /* predicted electron conc. */
    double pPred;                    /* predicted hole conc. */
    double uNet;                     /* net recombination rate */
    double dUdN;                     /* dU / dN */  
    double dUdP;                     /* dU / dP */  
    double dNdT;                     /* dN / dT */
    double dPdT;                     /* dP / dT */
    int nodeState;                   /* pointer to the state vector */
    unsigned evaluated : 1;          /* flag to indicated evaluated */
    /* sparse matrix pointers. pointers to doubles */
    /* DAG: diagonal pointers fXXiX1jX1 */
    double *fPsiPsiiM1;
    double *fPsiPsi;
    double *fPsiPsiiP1;
    double *fPsiPsijM1;
    double *fPsiPsijP1;
    double *fPsiN;
    double *fPsiP;
    double *fNPsiiM1;
    double *fNPsi;
    double *fNPsiiP1;
    double *fNPsijM1;
    double *fNPsijP1;
    double *fNPsiiM1jM1;
    double *fNPsiiM1jP1;
    double *fNPsiiP1jM1;
    double *fNPsiiP1jP1;
    double *fNNiM1;
    double *fNN;
    double *fNNiP1;
    double *fNNjM1;
    double *fNNjP1;
    double *fNNiM1jM1;
    double *fNNiM1jP1;
    double *fNNiP1jM1;
    double *fNNiP1jP1;
    double *fNP;
    double *fPPsiiM1;
    double *fPPsi;
    double *fPPsiiP1;
    double *fPPsijM1;
    double *fPPsijP1;
    double *fPPsiiM1jM1;
    double *fPPsiiM1jP1;
    double *fPPsiiP1jM1;
    double *fPPsiiP1jP1;
    double *fPPiM1;
    double *fPP;
    double *fPPiP1;
    double *fPPjM1;
    double *fPPjP1;
    double *fPPiM1jM1;
    double *fPPiM1jP1;
    double *fPPiP1jM1;
    double *fPPiP1jP1;
    double *fPN;
    /* DAG: Pointers for Surface-Field-Dependent Terms */
    /* For Oxide/Insulator on Silicon/Semiconductor:
                   Ox
      OxM1 + ----- + ----- + OxP1		OXIDE
           |       |       |
           |       |       |
    - InM1 + ----- + ----- + InP1 ---		INTERFACE
                   In       
    */
    double *fNPsiInM1;
    double *fNPsiIn;
    double *fNPsiInP1;
    double *fNPsiOxM1;
    double *fNPsiOx;
    double *fNPsiOxP1;
    double *fPPsiInM1;
    double *fPPsiIn;
    double *fPPsiInP1;
    double *fPPsiOxM1;
    double *fPPsiOx;
    double *fPPsiOxP1;
} TWOnode;

#define pTLElem pElems[0]
#define pTRElem pElems[1]
#define pBRElem pElems[2]
#define pBLElem pElems[3]

#define nodePsi   nodeState 
#define nodeN     nodeState+1
#define nodeP     nodeState+3

#define edgeDpsi  edgeState

#define TWOnumNodeStates 5
#define TWOnumEdgeStates 2

#endif
