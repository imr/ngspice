/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
Modified: 2000 AlansFixes
**********/

#ifndef SW
#define SW

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

    /* structures used to describe voltage controlled switches */


/* information to describe each instance */

typedef struct sSWinstance {
    struct sSWmodel *SWmodPtr;  /* backpointer to model */
    struct sSWinstance *SWnextInstance;   /* pointer to next instance of 
                                             * current model*/
    IFuid SWname;  /* pointer to character string naming this instance */
    int SWowner;  /* number of owner process */
    int SWstate;   /* pointer to start of switch's section of state vector */

    int SWposNode; /* number of positive node of switch */
    int SWnegNode; /* number of negative node of switch */
    int SWposCntrlNode; /* number of positive controlling node of switch */
    int SWnegCntrlNode; /* number of negative controlling node of switch */

    double *SWposPosptr;  /* pointer to sparse matrix diagonal at
                                (positive,positive) for switch conductance */
    double *SWnegPosptr;  /* pointer to sparse matrix offdiagonal at
                                (neagtive,positive) for switch conductance */
    double *SWposNegptr;  /* pointer to sparse matrix offdiagonal at
                                (positive,neagtive) for switch conductance */
    double *SWnegNegptr;  /* pointer to sparse matrix diagonal at
                                (neagtive,neagtive) for switch conductance */

    double SWcond;      /* conductance of the switch now */

    unsigned SWzero_stateGiven : 1;  /* flag to indicate initial state */
#ifndef NONOISE
    double SWnVar[NSTATVARS];
#else /* NONOISE */
    double *SWnVar;
#endif /* NONOISE */
} SWinstance ;

/* data per model */

#define SW_ON_CONDUCTANCE 1.0   /* default on conductance = 1 mho */
#define SW_OFF_CONDUCTANCE ckt->CKTgmin   /* default off conductance */
#define SW_NUM_STATES 2   

typedef struct sSWmodel {      /* model structure for a switch */
    int SWmodType;  /* type index of this device type */
    struct sSWmodel *SWnextModel; /* pointer to next possible model in 
                                     * linked list */
    SWinstance *SWinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid SWmodName;   /* pointer to character string naming this model */

    double SWonResistance;  /* switch "on" resistance */
    double SWoffResistance; /* switch "off" resistance */
    double SWvThreshold;    /* switching threshold voltage */
    double SWvHysteresis;   /* switching hysteresis voltage */
    double SWonConduct;     /* switch "on" conductance  */
    double SWoffConduct;    /* switch "off" conductance  */

    unsigned SWonGiven : 1;   /* flag to indicate on-resistance was specified */
    unsigned SWoffGiven : 1;  /* flag to indicate off-resistance was  "   */
    unsigned SWthreshGiven : 1; /* flag to indicate threshold volt was given */
    unsigned SWhystGiven : 1; /* flag to indicate hysteresis volt was given */
} SWmodel;

/* device parameters */
#define SW_IC_ON 1
#define SW_IC_OFF 2
#define SW_POS_NODE 3
#define SW_NEG_NODE 4
#define SW_POS_CONT_NODE 5
#define SW_NEG_CONT_NODE 6
#define SW_CURRENT 7
#define SW_POWER 8

/* model parameters */
#define SW_MOD_SW 101
#define SW_MOD_RON 102
#define SW_MOD_ROFF 103
#define SW_MOD_VTH 104
#define SW_MOD_VHYS 105
#define SW_MOD_GON 106
#define SW_MOD_GOFF 107

/* device questions */

/* model questions */

#include "swext.h"

#endif /*SW*/
