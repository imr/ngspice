/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef RES
#define RES

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

        /* definitions used to describe resistors */


/* information used to describe a single instance */

typedef struct sRESinstance {
    struct sRESmodel *RESmodPtr;            /* backpointer to model */
    struct sRESinstance *RESnextInstance;   /* pointer to next instance of 
                                             * current model*/

    IFuid RESname;      /* pointer to character string naming this instance */
    int RESowner;       /* number of owner process */
    int RESstate;       /* not used but needed for sructure consistency */
    int RESposNode;     /* number of positive node of resistor */
    int RESnegNode;     /* number of negative node of resistor */

    double REStemp;     /* temperature at which this resistor operates */
    double RESdtemp;    /* delta-temperature of a particular instance  */
    double RESconduct;  /* conductance at current analysis temperature */
    double RESresist;   /* resistance at temperature Tnom */
    double REScurrent;  /* The dc current in the resistor */
/* serban */
    double RESacResist;             /* AC resistance, useful for fancy .ac analyses */
    double RESacConduct;            /* AC conductance */
    double RESwidth;                /* width of the resistor */
    double RESlength;               /* length of the resistor */
    double RESscale;                /* Scale factor */
    double RESm;                    /* Multiplicity factor for this instance */
    double REStc1;                  /* first temperature coefficient of resistors */
    double REStc2;                  /* second temperature coefficient of resistors */
    int    RESnoisy;                /* Set if the resistor generates noise */
    double *RESposPosptr;           /* pointer to sparse matrix diagonal at 
                                     * (positive,positive) */
    double *RESnegNegptr;           /* pointer to sparse matrix diagonal at 
                                     * (negative,negative) */
    double *RESposNegptr;           /* pointer to sparse matrix offdiagonal at 
                                     * (positive,negative) */
    double *RESnegPosptr;           /* pointer to sparse matrix offdiagonal at 
                                     * (negative,positive) */
    unsigned RESresGiven    : 1;    /* flag to indicate resistance was specified */
    unsigned RESwidthGiven  : 1;    /* flag to indicate width given */
    unsigned RESlengthGiven : 1;    /* flag to indicate length given */
    unsigned RESscaleGiven  : 1;    /* flag to indicate scale given */
    unsigned REStempGiven   : 1;    /* indicates temperature specified */
    unsigned RESdtempGiven  : 1;    /* indicates delta-temp specified  */
/* serban */
    unsigned RESacresGiven  : 1;    /* indicates AC value specified */
    unsigned RESmGiven      : 1;    /* indicates M parameter specified */
    unsigned REStc1Given    : 1;    /* indicates tc1 parameter specified */
    unsigned REStc2Given    : 1;    /* indicates tc2 parameter specified */
    unsigned RESnoisyGiven  : 1;    /* indicates if noisy is specified */
    int    RESsenParmNo;            /* parameter # for sensitivity use;
                                     * set equal to  0 if not a design parameter*/

/* indices to array of RES noise sources */

#define RESTHNOIZ  0     /* Thermal noise source */
#define RESFLNOIZ  1     /* Flicker noise source */
#define RESTOTNOIZ 2     /* Total noise          */

#define RESNSRCS   3     /* the number of RES noise sources */


#ifndef NONOISE
    double RESnVar[NSTATVARS][RESNSRCS];
#else /* NONOISE */
    double **RESnVar;
#endif /* NONOISE */

} RESinstance ;


/* per model data */

typedef struct sRESmodel {       /* model structure for a resistor */
    int RESmodType; /* type index of this device type */
    struct sRESmodel *RESnextModel; /* pointer to next possible model in 
                                     * linked list */
    RESinstance * RESinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid RESmodName;       /* pointer to character string naming this model */

    double REStnom;         /* temperature at which resistance measured */
    double REStempCoeff1;   /* first temperature coefficient of resistors */
    double REStempCoeff2;   /* second temperature coefficient of resistors */
    double RESsheetRes;     /* sheet resistance of devices in ohms/square */
    double RESdefWidth;     /* default width of a resistor */
    double RESnarrow;       /* amount by which device is narrower than drawn */
    double RESshort;        /* amount by which device is shorter than drawn */
    double RESfNcoef;       /* Flicker noise coefficient */
    double RESfNexp;        /* Flicker noise exponent */
    unsigned REStnomGiven       :1; /* flag to indicate nominal temp. was given */
    unsigned REStc1Given        :1; /* flag to indicate tc1 was specified */
    unsigned REStc2Given        :1; /* flag to indicate tc2 was specified */
    unsigned RESsheetResGiven   :1; /* flag to indicate sheet resistance given*/
    unsigned RESdefWidthGiven   :1; /* flag to indicate default width given */
    unsigned RESnarrowGiven     :1; /* flag to indicate narrow effect given */
    unsigned RESshortGiven      :1; /* flag to indicate short effect given */
    unsigned RESfNcoefGiven     :1; /* flag to indicate kf given */
    unsigned RESfNexpGiven      :1; /* flag to indicate af given */
} RESmodel;

/* device parameters */
#define RES_RESIST 1
#define RES_WIDTH 2
#define RES_LENGTH 3
#define RES_CONDUCT 4
#define RES_RESIST_SENS 5
#define RES_CURRENT 6
#define RES_POWER 7
#define RES_TEMP 8
/* serban */
#define RES_ACRESIST 10
#define RES_ACCONDUCT 11
#define RES_M 12 /* pn */
#define RES_SCALE 13 /* pn */
#define RES_DTEMP 14 /* pn */
#define RES_NOISY 15 /* pn */
/* tanaka */
#define RES_TC1 16
#define RES_TC2 17

/* model parameters */
#define RES_MOD_TC1 101
#define RES_MOD_TC2 102
#define RES_MOD_RSH 103
#define RES_MOD_DEFWIDTH 104
#define RES_MOD_NARROW 105
#define RES_MOD_R 106
#define RES_MOD_TNOM 107
#define RES_MOD_SHORT 108
#define RES_MOD_KF 109
#define RES_MOD_AF 110

/* device questions */
#define RES_QUEST_SENS_REAL      201
#define RES_QUEST_SENS_IMAG      202
#define RES_QUEST_SENS_MAG       203
#define RES_QUEST_SENS_PH        204
#define RES_QUEST_SENS_CPLX      205
#define RES_QUEST_SENS_DC        206

/* model questions */

#include "resext.h"

#endif /*RES*/
