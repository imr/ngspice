/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef BALUN
#define BALUN


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /*
         * structures to describe Voltage Controlled Voltage Sources
         */

/* information to describe a single instance */

typedef struct sBALUNinstance {

    struct GENinstance gen;

#define BALUNmodPtr(inst) ((struct sBALUNmodel *)((inst)->gen.GENmodPtr))
#define BALUNnextInstance(inst) ((struct sBALUNinstance *)((inst)->gen.GENnextInstance))
#define BALUNname gen.GENname
#define BALUNstates gen.GENstate

    const int BALUNposNode;    /* number of positive node of source */
    const int BALUNnegNode;    /* number of negative node of source */
    const int BALUNcmNode;     /* cm node */
    const int BALUNdiffNode;   /* diff node */
    
    int BALUNbranchpos; /* branch for ipos */
    int BALUNbranchneg; /* branch for ineg */
    
    double *BALUNposIbrposPtr;
    double *BALUNnegIbrnegPtr;
    double *BALUNcmIbrposPtr;
    double *BALUNcmIbrnegPtr;
    double *BALUNdiffIbrposPtr;
    double *BALUNdiffIbrnegPtr;
    
    double *BALUNibrposDiffPtr;
    double *BALUNibrposPosPtr;
    double *BALUNibrposNegPtr;
    
    double *BALUNibrnegCmPtr;
    double *BALUNibrnegPosPtr;
    double *BALUNibrnegNegPtr;
    

    int  BALUNsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} BALUNinstance ;

/* per model data */

typedef struct sBALUNmodel {       /* model structure for a source */

    struct GENmodel gen;

#define BALUNmodType gen.GENmodType
#define BALUNnextModel(inst) ((struct sBALUNmodel *)((inst)->gen.GENnextModel))
#define BALUNinstances(inst) ((BALUNinstance *)((inst)->gen.GENinstances))
#define BALUNmodName gen.GENmodName

} BALUNmodel;

/* device parameters */
enum {
    BALUN_POS_NODE = 1,
    BALUN_NEG_NODE,
    BALUN_DIFF_NODE,
    BALUN_CM_NODE,
    BALUN_BRPOS,
    BALUN_BRNEG,
    BALUN_POWER,
};

/* model parameters */

/* device questions */
enum {
    BALUN_QUEST_NONE = 201,
};

/* model questions */

#include "balunext.h"

#endif /*BALUN*/
