/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#ifndef CAP
#define CAP


#include "ngspice/ifsim.h"
#include "ngspice/complex.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"

/* structures used to describe capacitors */


/* information to describe each instance */

typedef struct sCAPinstance {

    struct GENinstance gen;

#define CAPmodPtr(inst) ((struct sCAPmodel *)((inst)->gen.GENmodPtr))
#define CAPnextInstance(inst) ((struct sCAPinstance *)((inst)->gen.GENnextInstance))
#define CAPname gen.GENname
#define CAPstate gen.GENstate

    const int CAPposNode; /* number of positive node of capacitor */
    const int CAPnegNode; /* number of negative node of capacitor */

    double CAPtemp;     /* temperature at which this capacitor operates */
    double CAPdtemp;    /* delta-temperature of this instance */
    double CAPcapac;    /* capacitance */
    double CAPcapacinst;/* capacitance on instance line */
    double CAPinitCond; /* initial capacitor voltage if specified */
    double CAPwidth;    /* width of the capacitor */
    double CAPlength;   /* length of the capacitor */
    double CAPscale;    /* scale factor */
    double CAPm;        /* parallel multiplier */
    double CAPtc1;      /* first temperature coefficient of capacitors */
    double CAPtc2;      /* second temperature coefficient of capacitors */
    double CAPbv_max;   /* Maximum capacitor voltage */

    double *CAPposPosPtr;    /* pointer to sparse matrix diagonal at
                              * (positive,positive) */
    double *CAPnegNegPtr;    /* pointer to sparse matrix diagonal at
                              * (negative,negative) */
    double *CAPposNegPtr;    /* pointer to sparse matrix offdiagonal at
                              * (positive,negative) */
    double *CAPnegPosPtr;    /* pointer to sparse matrix offdiagonal at
                              * (negative,positive) */
    unsigned CAPcapGiven    : 1;   /* flag to indicate capacitance was specified */
    unsigned CAPicGiven     : 1;   /* flag to indicate init. cond. was specified */
    unsigned CAPwidthGiven  : 1;   /* flag to indicate capacitor width given */
    unsigned CAPlengthGiven : 1;   /* flag to indicate capacitor length given*/
    unsigned CAPtempGiven   : 1;   /* flag to indicate operating temp given */
    unsigned CAPdtempGiven  : 1;   /* flag to indicate delta temp given */
    unsigned CAPscaleGiven  : 1;   /* flag to indicate scale factor given */
    unsigned CAPmGiven      : 1;   /* flag to indicate parallel multiplier given */
    unsigned CAPtc1Given    : 1;    /* flag indicates tc1 was specified */
    unsigned CAPtc2Given    : 1;    /* flag indicates tc2 was specified */
    unsigned CAPbv_maxGiven : 1;    /* flags indicates maximum voltage is given */
    int    CAPsenParmNo;         /* parameter # for sensitivity use;
                set equal to  0 if not a design parameter*/

} CAPinstance ;

#define CAPqcap CAPstate    /* charge on the capacitor */
#define CAPccap CAPstate+1  /* current through the capacitor */

#define CAPnumStates 2

#define CAPsensxp CAPstate+2 /* charge sensitivities and their derivatives.
                              * +3 for the derivatives - pointer to the
                              * beginning of the array */

#define CAPnumSenStates 2


/* data per model */

typedef struct sCAPmodel {      /* model structure for a capacitor */

    struct GENmodel gen;

#define CAPmodType gen.GENmodType
#define CAPnextModel(inst) ((struct sCAPmodel *)((inst)->gen.GENnextModel))
#define CAPinstances(inst) ((CAPinstance *)((inst)->gen.GENinstances))
#define CAPmodName gen.GENmodName

    double CAPtnom;       /* temperature at which capacitance measured */
    double CAPtempCoeff1; /* linear temperature coefficient */
    double CAPtempCoeff2; /* quadratic temperature coefficient */
    double CAPmCap;       /* Model default capacitance */
    double CAPcj;         /* Unit Area Capacitance ( F/ M**2 ) */
    double CAPcjsw;       /* Unit Length Sidewall Capacitance ( F / M ) */
    double CAPdefWidth;   /* the default width of a capacitor */
    double CAPdefLength;  /* the default length of a capacitor */
    double CAPnarrow;     /* amount by which width are less than drawn */
    double CAPshort;      /* amount by which length are less than drawn */
    double CAPdel;        /* amount by which length and width are less than drawn */
    double CAPdi;         /* Relative dielectric constant */
    double CAPthick;      /* Insulator thickness */
    double CAPbv_max;     /* Maximum capacitor voltage */
    unsigned CAPmCapGiven      : 1;    /* flag indicates default capacitance given */
    unsigned CAPcjGiven        : 1;    /* Unit Area Capacitance ( F/ M**2 ) */
    unsigned CAPcjswGiven      : 1;    /* Unit Length Sidewall Capacitance( F/M )*/
    unsigned CAPdefWidthGiven  : 1;    /* flag indicates default width given*/
    unsigned CAPdefLengthGiven : 1;    /* flag indicates deafult lenght given */
    unsigned CAPnarrowGiven    : 1;    /* flag indicates narrowing factor given */
    unsigned CAPshortGiven     : 1;    /* flag indicates shortening factor given */
    unsigned CAPdelGiven       : 1;    /* flag indicates del factor given */
    unsigned CAPtnomGiven      : 1;    /* flag indicates nominal temp. given */
    unsigned CAPtc1Given       : 1;    /* flag indicates tc1 was specified */
    unsigned CAPtc2Given       : 1;    /* flag indicates tc2 was specified */
    unsigned CAPdiGiven        : 1;    /* flag indicates epsilon-ins given */
    unsigned CAPthickGiven     : 1;    /* flags indicates insulator thickness given */
    unsigned CAPbv_maxGiven    : 1;    /* flags indicates maximum voltage is given */

} CAPmodel;

/* device parameters */
enum {
    CAP_CAP = 1,
    CAP_IC,
    CAP_WIDTH,
    CAP_LENGTH,
    CAP_CAP_SENS,
    CAP_CURRENT,
    CAP_POWER,
    CAP_TEMP,
    CAP_DTEMP,
    CAP_SCALE,
    CAP_M,
    CAP_TC1,
    CAP_TC2,
    CAP_BV_MAX,
};

/* model parameters */
enum {
    CAP_MOD_CJ = 101,
    CAP_MOD_CJSW,
    CAP_MOD_DEFWIDTH,
    CAP_MOD_C,
    CAP_MOD_NARROW,
    CAP_MOD_SHORT,
    CAP_MOD_DEL,
    CAP_MOD_TC1,
    CAP_MOD_TC2,
    CAP_MOD_TNOM,
    CAP_MOD_DI,
    CAP_MOD_THICK,
    CAP_MOD_CAP,
    CAP_MOD_DEFLENGTH,
    CAP_MOD_BV_MAX,
};

/* device questions */
enum {
    CAP_QUEST_SENS_REAL = 201,
    CAP_QUEST_SENS_IMAG,
    CAP_QUEST_SENS_MAG,
    CAP_QUEST_SENS_PH,
    CAP_QUEST_SENS_CPLX,
    CAP_QUEST_SENS_DC,
};

#include "capext.h"

#endif /*CAP*/
