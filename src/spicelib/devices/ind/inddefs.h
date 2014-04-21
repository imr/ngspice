/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef IND
#define IND


/* turn on mutual inductor code */
#define MUTUAL

#include "ngspice/ifsim.h"
#include "ngspice/complex.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"

typedef struct sINDinstance INDinstance;
typedef struct sINDmodel INDmodel;
typedef struct sMUTinstance MUTinstance;
typedef struct sMUTmodel MUTmodel;


/* structures used to descrive inductors */


/* information needed for each instance */

struct sINDinstance {

    struct GENinstance gen;

#define INDmodPtr(inst) ((INDmodel *)((inst)->gen.GENmodPtr))
#define INDnextInstance(inst) ((INDinstance *)((inst)->gen.GENnextInstance))
#define INDname gen.GENname
#define INDstate gen.GENstate

    const int INDposNode; /* number of positive node of inductor */
    const int INDnegNode; /* number of negative node of inductor */

    int INDbrEq;    /* number of the branch equation added for current */
    double INDinduct;    /* inductance */
    double INDm;         /* Parallel multiplier */
    double INDtc1;       /* first temperature coefficient of resistors */
    double INDtc2;       /* second temperature coefficient of resistors */
    double INDtemp;      /* Instance operating temperature */
    double INDdtemp;     /* Delta temp. of instance */
    double INDscale;     /* Scale factor */
    double INDnt;        /* Number of turns */
    double INDinitCond;  /* initial inductor voltage if specified */

    double *INDposIbrPtr;    /* pointer to sparse matrix diagonal at
                              * (positive,branch eq) */
    double *INDnegIbrPtr;    /* pointer to sparse matrix diagonal at
                              * (negative,branch eq) */
    double *INDibrNegPtr;    /* pointer to sparse matrix offdiagonal at
                              * (branch eq,negative) */
    double *INDibrPosPtr;    /* pointer to sparse matrix offdiagonal at
                              * (branch eq,positive) */
    double *INDibrIbrPtr;    /* pointer to sparse matrix offdiagonal at
                              * (branch eq,branch eq) */

    unsigned INDindGiven   : 1;   /* flag to indicate inductance was specified */
    unsigned INDicGiven    : 1;   /* flag to indicate init. cond. was specified */
    unsigned INDmGiven     : 1;   /* flag to indicate multiplier given */
    unsigned INDtc1Given   : 1;   /* indicates tc1 parameter specified */
    unsigned INDtc2Given   : 1;   /* indicates tc2 parameter specified */
    unsigned INDtempGiven  : 1;   /* flag to indicate operating temp. given */
    unsigned INDdtempGiven : 1;   /* flag to indicate delta temp. given */
    unsigned INDscaleGiven : 1;   /* flag to indicate scale factor given */
    unsigned INDntGiven    : 1;   /* flag to indicate number of turns given */
    int  INDsenParmNo;       /* parameter # for sensitivity use;
                              * set equal to  0 if not a design parameter */

    struct INDsystem *system;
    INDinstance *system_next_ind;
    int system_idx;

#ifdef KLU
    BindElement *INDposIbrBinding ;
    BindElement *INDnegIbrBinding ;
    BindElement *INDibrNegBinding ;
    BindElement *INDibrPosBinding ;
    BindElement *INDibrIbrBinding ;
#endif
};

#define INDflux INDstate     /* flux in the inductor */
#define INDvolt INDstate+1   /* voltage - save an entry in table */
#define INDsensxp INDstate+2 /* charge sensitivities and their derivatives.
                              *  +3 for the derivatives - pointer to the
                              *  beginning of the array */


/* per model data */

struct sINDmodel {             /* model structure for an inductor */

    struct GENmodel gen;

#define INDmodType gen.GENmodType
#define INDnextModel(inst) ((INDmodel *)((inst)->gen.GENnextModel))
#define INDinstances(inst) ((INDinstance *)((inst)->gen.GENinstances))
#define INDmodName gen.GENmodName

    double INDmInd;        /* Model inductance */
    double INDtnom;        /* temperature at which inductance measured */
    double INDtempCoeff1;  /* first temperature coefficient */
    double INDtempCoeff2;  /* second  temperature coefficient */
    double INDcsect;       /* Cross section of inductor */
    double INDlength;      /* Mean length of magnetic path */
    double INDmodNt;       /* Model number of turns */
    double INDmu;          /* Relative magnetic permeability */

    unsigned INDtnomGiven  : 1; /* flag to indicate nominal temp was given */
    unsigned INDtc1Given   : 1; /* flag to indicate tc1 was specified */
    unsigned INDtc2Given   : 1; /* flag to indicate tc2 was specified */
    unsigned INDcsectGiven : 1; /* flag to indicate cross section given */
    unsigned INDlengthGiven: 1; /* flag to indicate length given */
    unsigned INDmodNtGiven : 1; /* flag to indicate mod. n. of turns given */
    unsigned INDmuGiven    : 1; /* flag to indicate mu_r given */
    unsigned INDmIndGiven  : 1; /* flag to indicate model inductance given */

    double INDspecInd;     /* Specific (one turn) inductance */
};


#ifdef MUTUAL

/* structures used to describe mutual inductors */


/* information needed for each instance */

struct sMUTinstance {

    struct GENinstance gen;

#define MUTmodPtr(inst) ((MUTmodel *)((inst)->gen.GENmodPtr))
#define MUTnextInstance(inst) ((MUTinstance *)((inst)->gen.GENnextInstance))
#define MUTname gen.GENname
#define MUTstates gen.GENstate

    double MUTcoupling;   /* mutual inductance input by user */
    double MUTfactor;     /* mutual inductance scaled for internal use */
    IFuid MUTindName1;    /* name of coupled inductor 1 */
    IFuid MUTindName2;    /* name of coupled inductor 2 */
    INDinstance *MUTind1; /* pointer to coupled inductor 1 */
    INDinstance *MUTind2; /* pointer to coupled inductor 2 */
    double *MUTbr1br2Ptr;    /* pointers to off-diagonal intersections of */
    double *MUTbr2br1Ptr;    /* current branch equations in matrix */

    unsigned MUTindGiven : 1;   /* flag to indicate inductance was specified */
    int  MUTsenParmNo;          /* parameter # for sensitivity use;
                                 * set equal to  0 if not a design parameter */

    MUTinstance *system_next_mut;

#ifdef KLU
    BindElement *MUTbr1br2Binding ;
    BindElement *MUTbr2br1Binding ;
#endif
};


/* per model data */

struct sMUTmodel {             /* model structure for a mutual inductor */

    struct GENmodel gen;

#define MUTmodType gen.GENmodType
#define MUTnextModel(inst) ((MUTmodel *)((inst)->gen.GENnextModel))
#define MUTinstances(inst) ((MUTinstance *)((inst)->gen.GENinstances))
#define MUTmodName gen.GENmodName

};


struct INDsystem {
    int size;
    INDinstance *first_ind;
    MUTinstance *first_mut;
    struct INDsystem *next_system;
};

#endif /*MUTUAL*/

/* device parameters */
#define IND_IND      1
#define IND_IC       2
#define IND_FLUX     3
#define IND_VOLT     4
#define IND_IND_SENS 5
#define IND_CURRENT  6
#define IND_POWER    7
#define IND_M        8
#define IND_TEMP     9
#define IND_DTEMP   10
#define IND_SCALE   11
#define IND_NT      12
#define IND_TC1     13
#define IND_TC2     14

/* model parameters */
#define IND_MOD_IND    100
#define IND_MOD_TC1    101
#define IND_MOD_TC2    102
#define IND_MOD_TNOM   103
#define IND_MOD_CSECT  104
#define IND_MOD_LENGTH 105
#define IND_MOD_NT     106
#define IND_MOD_MU     107
#define IND_MOD_L      108

/* device questions */
#define IND_QUEST_SENS_REAL      201
#define IND_QUEST_SENS_IMAG      202
#define IND_QUEST_SENS_MAG       203
#define IND_QUEST_SENS_PH        204
#define IND_QUEST_SENS_CPLX      205
#define IND_QUEST_SENS_DC        206

#ifdef MUTUAL
/* device parameters */
#define MUT_COEFF       401
#define MUT_IND1        402
#define MUT_IND2        403
#define MUT_COEFF_SENS  404

/* model parameters */

/* device questions */
#define MUT_QUEST_SENS_REAL      601
#define MUT_QUEST_SENS_IMAG      602
#define MUT_QUEST_SENS_MAG       603
#define MUT_QUEST_SENS_PH        604
#define MUT_QUEST_SENS_CPLX      605
#define MUT_QUEST_SENS_DC        606

#endif /*MUTUAL*/

#include "indext.h"

#endif /*IND*/
