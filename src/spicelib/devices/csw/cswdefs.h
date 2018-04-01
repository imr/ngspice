/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
Modified: 2000 AlansFixes
**********/

#ifndef CSW
#define CSW

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/noisedef.h"
#include "ngspice/complex.h"

    /* structures used to describe current controlled switches */


/* information to describe each instance */

typedef struct sCSWinstance {

    struct GENinstance gen;

#define CSWmodPtr(inst) ((struct sCSWmodel *)((inst)->gen.GENmodPtr))
#define CSWnextInstance(inst) ((struct sCSWinstance *)((inst)->gen.GENnextInstance))
#define CSWname gen.GENname
#define CSWstate gen.GENstate

    const int CSWposNode; /* number of positive node of switch */
    const int CSWnegNode; /* number of negative node of switch */
    int CSWcontBranch; /* number of branch of controlling current */

    IFuid CSWcontName; /* name of controlling source */

    double *CSWposPosPtr;  /* pointer to sparse matrix diagonal at
                                (positive,positive) for switch conductance */
    double *CSWnegPosPtr;  /* pointer to sparse matrix offdiagonal at
                                (neagtive,positive) for switch conductance */
    double *CSWposNegPtr;  /* pointer to sparse matrix offdiagonal at
                                (positive,neagtive) for switch conductance */
    double *CSWnegNegPtr;  /* pointer to sparse matrix diagonal at
                                (neagtive,neagtive) for switch conductance */

    double CSWcond;     /* current conductance of switch */

    unsigned CSWzero_stateGiven : 1;  /* flag to indicate initial state */
#ifndef NONOISE
    double CSWnVar[NSTATVARS];
#else /* NONOISE */
	double *CSWnVar;
#endif /* NONOISE */

#ifdef KLU
    BindElement *CSWposPosBinding ;
    BindElement *CSWposNegBinding ;
    BindElement *CSWnegPosBinding ;
    BindElement *CSWnegNegBinding ;
#endif

} CSWinstance ;

/* data per model */

#define CSW_ON_CONDUCTANCE 1.0   /* default on conductance = 1 mho */
#define CSW_OFF_CONDUCTANCE ckt->CKTgmin   /* default off conductance */
#define CSW_NUM_STATES 2   

typedef struct sCSWmodel {      /* model structure for a switch */

    struct GENmodel gen;

#define CSWmodType gen.GENmodType
#define CSWnextModel(inst) ((struct sCSWmodel *)((inst)->gen.GENnextModel))
#define CSWinstances(inst) ((CSWinstance *)((inst)->gen.GENinstances))
#define CSWmodName gen.GENmodName

    double CSWonResistance;  /* switch "on" resistance */
    double CSWoffResistance; /* switch "off" resistance */
    double CSWiThreshold;    /* switching threshold current */
    double CSWiHysteresis;   /* switching hysteresis current */
    double CSWonConduct;     /* switch "on" conductance  */
    double CSWoffConduct;    /* switch "off" conductance  */

    unsigned CSWonGiven : 1; /* flag to indicate on-resistance was specified */
    unsigned CSWoffGiven : 1;/* flag to indicate off-resistance was  "   */
    unsigned CSWthreshGiven : 1;/* flag to indicate threshold volt was given */
    unsigned CSWhystGiven : 1; /* flag to indicate hysteresis volt was given */
} CSWmodel;

/* device parameters */
#define CSW_CONTROL 1
#define CSW_IC_ON 2
#define CSW_IC_OFF 3
#define CSW_POS_NODE 4
#define CSW_NEG_NODE 5
#define CSW_CURRENT 6
#define CSW_POWER 7

/* model parameters */
#define CSW_CSW 101
#define CSW_RON 102
#define CSW_ROFF 103
#define CSW_ITH 104
#define CSW_IHYS 105
#define CSW_GON 106
#define CSW_GOFF 107

/* device questions */

/* model questions */

#include "cswext.h"

#endif /*CSW*/
