/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef TRA
#define TRA


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /* definitions used to describe transmission liness */


/* information used to describe a single instance */

typedef struct sTRAinstance {

    struct GENinstance gen;

#define TRAmodPtr(inst) ((struct sTRAmodel *)((inst)->gen.GENmodPtr))
#define TRAnextInstance(inst) ((struct sTRAinstance *)((inst)->gen.GENnextInstance))
#define TRAname gen.GENname
#define TRAstate gen.GENstate

    const int TRAposNode1;    /* number of positive node of end 1 of t. line */
    const int TRAnegNode1;    /* number of negative node of end 1 of t. line */
    const int TRAposNode2;    /* number of positive node of end 2 of t. line */
    const int TRAnegNode2;    /* number of negative node of end 2 of t. line */
    int TRAintNode1;    /* number of internal node of end 1 of t. line */
    int TRAintNode2;    /* number of internal node of end 2 of t. line */

    double TRAimped;    /* impedance  - input */
    double TRAconduct;  /* conductance  - calculated */
    double TRAtd;       /* propagation delay */
    double TRAnl;       /* normalized length */
    double TRAf;        /* frequency at which nl is measured */
    double TRAinput1;   /* accumulated excitation for port 1 */
    double TRAinput2;   /* accumulated excitation for port 2 */
    double TRAinitVolt1;    /* initial condition:  voltage on port 1 */
    double TRAinitCur1;     /* initial condition:  current at port 1 */
    double TRAinitVolt2;    /* initial condition:  voltage on port 2 */
    double TRAinitCur2;     /* initial condition:  current at port 2 */
    double TRAreltol;       /* relative deriv. tol. for breakpoint setting */
    double TRAabstol;       /* absolute deriv. tol. for breakpoint setting */
    double *TRAdelays;     /* delayed values of excitation */
    int TRAsizeDelay;       /* size of active delayed table */
    int TRAallocDelay;      /* allocated size of delayed table */
    int TRAbrEq1;       /* number of branch equation for end 1 of t. line */
    int TRAbrEq2;       /* number of branch equation for end 2 of t. line */
    double *TRAibr1Ibr2Ptr;     /* pointer to sparse matrix */
    double *TRAibr1Int1Ptr;     /* pointer to sparse matrix */
    double *TRAibr1Neg1Ptr;     /* pointer to sparse matrix */
    double *TRAibr1Neg2Ptr;     /* pointer to sparse matrix */
    double *TRAibr1Pos2Ptr;     /* pointer to sparse matrix */
    double *TRAibr2Ibr1Ptr;     /* pointer to sparse matrix */
    double *TRAibr2Int2Ptr;     /* pointer to sparse matrix */
    double *TRAibr2Neg1Ptr;     /* pointer to sparse matrix */
    double *TRAibr2Neg2Ptr;     /* pointer to sparse matrix */
    double *TRAibr2Pos1Ptr;     /* pointer to sparse matrix */
    double *TRAint1Ibr1Ptr;     /* pointer to sparse matrix */
    double *TRAint1Int1Ptr;     /* pointer to sparse matrix */
    double *TRAint1Pos1Ptr;     /* pointer to sparse matrix */
    double *TRAint2Ibr2Ptr;     /* pointer to sparse matrix */
    double *TRAint2Int2Ptr;     /* pointer to sparse matrix */
    double *TRAint2Pos2Ptr;     /* pointer to sparse matrix */
    double *TRAneg1Ibr1Ptr;     /* pointer to sparse matrix */
    double *TRAneg2Ibr2Ptr;     /* pointer to sparse matrix */
    double *TRApos1Int1Ptr;     /* pointer to sparse matrix */
    double *TRApos1Pos1Ptr;     /* pointer to sparse matrix */
    double *TRApos2Int2Ptr;     /* pointer to sparse matrix */
    double *TRApos2Pos2Ptr;     /* pointer to sparse matrix */
    unsigned TRAimpedGiven : 1; /* flag to indicate impedence was specified */
    unsigned TRAtdGiven : 1;    /* flag to indicate delay was specified */
    unsigned TRAnlGiven : 1;    /* flag to indicate norm length was specified */
    unsigned TRAfGiven : 1;     /* flag to indicate freq was specified */
    unsigned TRAicV1Given : 1;  /* flag to ind. init. voltage at port 1 given */
    unsigned TRAicC1Given : 1;  /* flag to ind. init. current at port 1 given */
    unsigned TRAicV2Given : 1;  /* flag to ind. init. voltage at port 2 given */
    unsigned TRAicC2Given : 1;  /* flag to ind. init. current at port 2 given */
    unsigned TRAreltolGiven:1;  /* flag to ind. relative deriv. tol. given */
    unsigned TRAabstolGiven:1;  /* flag to ind. absolute deriv. tol. given */
} TRAinstance ;


/* per model data */

typedef struct sTRAmodel {       /* model structure for a transmission lines */

    struct GENmodel gen;

#define TRAmodType gen.GENmodType
#define TRAnextModel(inst) ((struct sTRAmodel *)((inst)->gen.GENnextModel))
#define TRAinstances(inst) ((TRAinstance *)((inst)->gen.GENinstances))
#define TRAmodName gen.GENmodName

} TRAmodel;

/* device parameters */
enum {
    TRA_Z0 = 1,
    TRA_TD,
    TRA_NL,
    TRA_FREQ,
    TRA_V1,
    TRA_I1,
    TRA_V2,
    TRA_I2,
    TRA_IC,
    TRA_RELTOL,
    TRA_ABSTOL,
    TRA_POS_NODE1,
    TRA_NEG_NODE1,
    TRA_POS_NODE2,
    TRA_NEG_NODE2,
    TRA_INPUT1,
    TRA_INPUT2,
    TRA_DELAY,
    TRA_BR_EQ1,
    TRA_BR_EQ2,
    TRA_INT_NODE1,
    TRA_INT_NODE2,
};

/* model parameters */

/* device questions */

/* model questions */

#include "traext.h"

#endif /*TRA*/

