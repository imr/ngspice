/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef IND
#define IND


/* turn on mutual inductor code */
#define MUTUAL

#include "ifsim.h"
#include "complex.h"
#include "gendefs.h"
#include "cktdefs.h"

        /* structures used to descrive inductors */


/* information needed for each instance */

typedef struct sINDinstance {
    struct sINDmodel *INDmodPtr;    /* backpointer to model */
    struct sINDinstance *INDnextInstance;   /* pointer to next instance of 
                                             * current model*/
    IFuid INDname;  /* pointer to character string naming this instance */
    int INDowner;  /* number of owner process */
    int INDstate;   /* pointer to beginning of state vector for inductor */
    int INDposNode; /* number of positive node of inductor */
    int INDnegNode; /* number of negative node of inductor */

    int INDbrEq;    /* number of the branch equation added for current */
    double INDinduct;    /* inductance */
    double INDinitCond; /* initial inductor voltage if specified */

    double *INDposIbrptr;    /* pointer to sparse matrix diagonal at 
                              * (positive,branch eq) */
    double *INDnegIbrptr;    /* pointer to sparse matrix diagonal at 
                              * (negative,branch eq) */
    double *INDibrNegptr;    /* pointer to sparse matrix offdiagonal at 
                              * (branch eq,negative) */
    double *INDibrPosptr;    /* pointer to sparse matrix offdiagonal at 
                              * (branch eq,positive) */
    double *INDibrIbrptr;    /* pointer to sparse matrix offdiagonal at 
                              * (branch eq,branch eq) */
    unsigned INDindGiven : 1;   /* flag to indicate inductance was specified */
    unsigned INDicGiven  : 1;   /* flag to indicate init. cond. was specified */
    int  INDsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} INDinstance ;

#define INDflux INDstate    /* flux in the inductor */
#define INDvolt INDstate+1  /* voltage - save an entry in table */
#define INDsensxp INDstate+2 /* charge sensitivities and their derivatives.
                                +3 for the derivatives - pointer to the
                beginning of the array */


/* per model data */

typedef struct sINDmodel {       /* model structure for an inductor */
    int INDmodType; /* type index of this device type */
    struct sINDmodel *INDnextModel; /* pointer to next possible model in 
                                     * linked list */
    INDinstance * INDinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid INDmodName;       /* pointer to character string naming this model */
} INDmodel;


#ifdef MUTUAL

        /* structures used to describe mutual inductors */


/* information needed for each instance */

typedef struct sMUTinstance {
    struct sMUTmodel *MUTmodPtr;    /* backpointer to model */
    struct sMUTinstance *MUTnextInstance;   /* pointer to next instance of 
                                             * current model*/
    IFuid MUTname;  /* pointer to character string naming this instance */
    int MUTowner;  /* number of owner process */
    double MUTcoupling;     /* mutual inductance input by user */
    double MUTfactor;       /* mutual inductance scaled for internal use */
    IFuid MUTindName1;  /* name of coupled inductor 1 */
    IFuid MUTindName2;  /* name of coupled inductor 2 */
    INDinstance *MUTind1;   /* pointer to coupled inductor 1 */
    INDinstance *MUTind2;   /* pointer to coupled inductor 2 */
    double *MUTbr1br2;  /* pointers to off-diagonal intersections of */
    double *MUTbr2br1;  /* current branch equations in matrix */

    unsigned MUTindGiven : 1;   /* flag to indicate inductance was specified */
    int  MUTsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/


} MUTinstance ;


/* per model data */

typedef struct sMUTmodel {       /* model structure for a mutual inductor */
    int MUTmodType; /* type index of this device type */
    struct sMUTmodel *MUTnextModel; /* pointer to next possible model in 
                                     * linked list */
    MUTinstance * MUTinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid MUTmodName;       /* pointer to character string naming this model */
} MUTmodel;

#endif /*MUTUAL*/

/* device parameters */
#define IND_IND 1
#define IND_IC 2
#define IND_FLUX 3
#define IND_VOLT 4
#define IND_IND_SENS 5
#define IND_CURRENT 6
#define IND_POWER 7

/* model parameters */

/* device questions */
#define IND_QUEST_SENS_REAL      201
#define IND_QUEST_SENS_IMAG      202
#define IND_QUEST_SENS_MAG       203
#define IND_QUEST_SENS_PH        204
#define IND_QUEST_SENS_CPLX      205
#define IND_QUEST_SENS_DC        206

#ifdef MUTUAL
/* device parameters */
#define MUT_COEFF 401
#define MUT_IND1 402
#define MUT_IND2 403
#define MUT_COEFF_SENS 404

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
