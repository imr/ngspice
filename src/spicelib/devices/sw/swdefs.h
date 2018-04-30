/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
Modified: 2000 AlansFixes
**********/

#ifndef SW
#define SW

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* structures used to describe voltage controlled switches */


/* information to describe each instance */

typedef struct sSWinstance {

    struct GENinstance gen;

#define SWmodPtr(inst) ((struct sSWmodel *)((inst)->gen.GENmodPtr))
#define SWnextInstance(inst) ((struct sSWinstance *)((inst)->gen.GENnextInstance))
#define SWname gen.GENname
#define SWstate gen.GENstate

    const int SWposNode;      /* number of positive node of switch */
    const int SWnegNode;      /* number of negative node of switch */
    const int SWposCntrlNode; /* number of positive controlling node of switch */
    const int SWnegCntrlNode; /* number of negative controlling node of switch */

    double *SWposPosPtr;  /* pointer to sparse matrix diagonal at
                             (positive,positive) for switch conductance */
    double *SWnegPosPtr;  /* pointer to sparse matrix offdiagonal at
                             (neagtive,positive) for switch conductance */
    double *SWposNegPtr;  /* pointer to sparse matrix offdiagonal at
                             (positive,neagtive) for switch conductance */
    double *SWnegNegPtr;  /* pointer to sparse matrix diagonal at
                             (neagtive,neagtive) for switch conductance */

    double SWcond;        /* conductance of the switch now */

    unsigned SWzero_stateGiven : 1;  /* flag to indicate initial state */
#ifndef NONOISE
    double SWnVar[NSTATVARS];
#else
    double *SWnVar;
#endif
} SWinstance;

/* data per model */

#define SW_ON_CONDUCTANCE 1.0   /* default on conductance = 1 mho */
#define SW_OFF_CONDUCTANCE ckt->CKTgmin   /* default off conductance */
#define SW_NUM_STATES 2

#define SWswitchstate SWstate+0
#define SWctrlvalue   SWstate+1


typedef struct sSWmodel {      /* model structure for a switch */

    struct GENmodel gen;

#define SWmodType gen.GENmodType
#define SWnextModel(inst) ((struct sSWmodel *)((inst)->gen.GENnextModel))
#define SWinstances(inst) ((SWinstance *) ((inst)->gen.GENinstances))
#define SWmodName gen.GENmodName

    double SWonResistance;  /* switch "on" resistance */
    double SWoffResistance; /* switch "off" resistance */
    double SWvThreshold;    /* switching threshold voltage */
    double SWvHysteresis;   /* switching hysteresis voltage */
    double SWonConduct;     /* switch "on" conductance  */
    double SWoffConduct;    /* switch "off" conductance  */

    unsigned SWonGiven : 1;     /* flag to indicate on-resistance was specified */
    unsigned SWoffGiven : 1;    /* flag to indicate off-resistance was  "   */
    unsigned SWthreshGiven : 1; /* flag to indicate threshold volt was given */
    unsigned SWhystGiven : 1;   /* flag to indicate hysteresis volt was given */
} SWmodel;

/* device parameters */
enum {
    SW_IC_ON = 1,
    SW_IC_OFF,
    SW_POS_NODE,
    SW_NEG_NODE,
    SW_POS_CONT_NODE,
    SW_NEG_CONT_NODE,
    SW_CURRENT,
    SW_POWER,
};

/* model parameters */
enum {
    SW_MOD_SW = 101,
    SW_MOD_RON,
    SW_MOD_ROFF,
    SW_MOD_VTH,
    SW_MOD_VHYS,
    SW_MOD_GON,
    SW_MOD_GOFF,
};

/* device questions */

/* model questions */

#include "swext.h"

#endif
