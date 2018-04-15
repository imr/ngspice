/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef RES
#define RES

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* definitions used to describe resistors */

/* indices to array of RES noise sources */

enum {
    RESTHNOIZ = 0,  /* Thermal noise source */
    RESFLNOIZ,      /* Flicker noise source */
    RESTOTNOIZ,     /* Total noise          */
    /* finally, the number of noise sources */
    RESNSRCS
};

/* information used to describe a single instance */

typedef struct sRESinstance {

    struct GENinstance gen;

#define RESmodPtr(inst) ((struct sRESmodel *)((inst)->gen.GENmodPtr))
#define RESnextInstance(inst) ((struct sRESinstance *)((inst)->gen.GENnextInstance))
#define RESname gen.GENname
#define RESstate gen.GENstate

    const int RESposNode;     /* number of positive node of resistor */
    const int RESnegNode;     /* number of negative node of resistor */

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
    double REStce;                  /* exponential temperature coefficient of resistors */
    double RESbv_max;               /* Maximum resistor voltage */
    int    RESnoisy;                /* Set if the resistor generates noise */
    double RESeffNoiseArea;         /* effective resistor area for noise calculation */
    double *RESposPosPtr;           /* pointer to sparse matrix diagonal at
                                     * (positive,positive) */
    double *RESnegNegPtr;           /* pointer to sparse matrix diagonal at
                                     * (negative,negative) */
    double *RESposNegPtr;           /* pointer to sparse matrix offdiagonal at
                                     * (positive,negative) */
    double *RESnegPosPtr;           /* pointer to sparse matrix offdiagonal at
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
    unsigned REStceGiven    : 1;    /* indicates tce parameter specified */
    unsigned RESnoisyGiven  : 1;    /* indicates if noisy is specified */
    unsigned RESbv_maxGiven : 1;    /* flags indicates maximum voltage is given */
    int    RESsenParmNo;            /* parameter # for sensitivity use;
                                     * set equal to  0 if not a design parameter*/

#ifndef NONOISE
    double RESnVar[NSTATVARS][RESNSRCS];
#else /* NONOISE */
    double **RESnVar;
#endif /* NONOISE */

} RESinstance ;


/* per model data */

typedef struct sRESmodel {       /* model structure for a resistor */

    struct GENmodel gen;

#define RESmodType gen.GENmodType
#define RESnextModel(inst) ((struct sRESmodel *)((inst)->gen.GENnextModel))
#define RESinstances(inst) ((RESinstance *)((inst)->gen.GENinstances))
#define RESmodName gen.GENmodName

    double REStnom;         /* temperature at which resistance measured */
    double REStempCoeff1;   /* first temperature coefficient of resistors */
    double REStempCoeff2;   /* second temperature coefficient of resistors */
    double REStempCoeffe;   /* exponential temperature coefficient of resistors */
    double RESsheetRes;     /* sheet resistance of devices in ohms/square */
    double RESdefWidth;     /* default width of a resistor */
    double RESdefLength;    /* default length of a resistor */
    double RESnarrow;       /* amount by which device is narrower than drawn */
    double RESshort;        /* amount by which device is shorter than drawn */
    double RESfNcoef;       /* Flicker noise coefficient */
    double RESfNexp;        /* Flicker noise exponent */
    double RESres;          /* Default model resistance */
    double RESbv_max;       /* Maximum resistor voltage */
    double RESlf;           /* length exponent for noise calculation */
    double RESwf;           /* width exponent for noise calculation */
    double RESef;           /* frequncy exponent for noise calculation */
    unsigned REStnomGiven       :1; /* flag to indicate nominal temp. was given */
    unsigned REStc1Given        :1; /* flag to indicate tc1 was specified */
    unsigned REStc2Given        :1; /* flag to indicate tc2 was specified */
    unsigned REStceGiven        :1; /* flag to indicate tce was specified */
    unsigned RESsheetResGiven   :1; /* flag to indicate sheet resistance given*/
    unsigned RESdefWidthGiven   :1; /* flag to indicate default width given */
    unsigned RESdefLengthGiven  :1; /* flag to indicate default length given */
    unsigned RESnarrowGiven     :1; /* flag to indicate narrow effect given */
    unsigned RESshortGiven      :1; /* flag to indicate short effect given */
    unsigned RESfNcoefGiven     :1; /* flag to indicate kf given */
    unsigned RESfNexpGiven      :1; /* flag to indicate af given */
    unsigned RESresGiven        :1; /* flag to indicate model resistance given */
    unsigned RESbv_maxGiven     :1; /* flags indicates maximum voltage is given */
    unsigned RESlfGiven         :1; /* flags indicates lf is given */
    unsigned RESwfGiven         :1; /* flags indicates wf is given */
    unsigned RESefGiven         :1; /* flags indicates ef is given */
} RESmodel;

/* device parameters */
enum {
    RES_RESIST = 1,
    RES_WIDTH,
    RES_LENGTH,
    RES_CONDUCT,
    RES_RESIST_SENS,
    RES_CURRENT,
    RES_POWER,
    RES_TEMP,
};

/* serban */
enum {
    RES_ACRESIST = 10,
    RES_ACCONDUCT,
    RES_M,
    RES_SCALE,
    RES_DTEMP,
    RES_NOISY,
    RES_TC1,
    RES_TC2,
    RES_BV_MAX,
    RES_TCE,
};

/* model parameters */
enum {
    RES_MOD_TC1 = 101,
    RES_MOD_TC2,
    RES_MOD_RSH,
    RES_MOD_DEFWIDTH,
    RES_MOD_DEFLENGTH,
    RES_MOD_NARROW,
    RES_MOD_R,
    RES_MOD_TNOM,
    RES_MOD_SHORT,
    RES_MOD_KF,
    RES_MOD_AF,
    RES_MOD_BV_MAX,
    RES_MOD_LF,
    RES_MOD_WF,
    RES_MOD_EF,
    RES_MOD_TCE,
};

/* device questions */
enum {
    RES_QUEST_SENS_REAL = 201,
    RES_QUEST_SENS_IMAG,
    RES_QUEST_SENS_MAG,
    RES_QUEST_SENS_PH,
    RES_QUEST_SENS_CPLX,
    RES_QUEST_SENS_DC,
};

/* model questions */

#include "resext.h"

extern void RESupdate_conduct(RESinstance *, bool spill_warnings);

#endif /*RES*/
