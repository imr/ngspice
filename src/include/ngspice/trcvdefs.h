/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/
/*
 */
#ifndef ngspice_TRCVDEFS_H
#define ngspice_TRCVDEFS_H


#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/gendefs.h"
    /*
     * structures used to describe D.C. transfer curve analyses to
     * be performed.
     */

#define TRCVNESTLEVEL 2 /* depth of nesting of curves - 2 for spice2 */

/* PN: The following define is for temp sweep */
/* Courtesy of: Serban M. Popescu */
#ifndef TEMP_CODE
#define TEMP_CODE 1023
#endif

/* HSPICE-compat sweep of a `.param` name (instead of a V/I source,
 * resistor, or temperature).  Used by `.dc paramName start stop step`
 * where paramName matches an existing .param.  Each step updates the
 * numparam entry's value and pushes it to any V/I source whose DC
 * field was bound at parse time (see inpdpar.c
 * dpar_register_binding). */
#ifndef PARAM_CODE
#define PARAM_CODE 1022
#endif

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
    double TRCVvStart[TRCVNESTLEVEL];   /* starting voltage/current */
    double TRCVvStop[TRCVNESTLEVEL];    /* ending voltage/current */
    double TRCVvStep[TRCVNESTLEVEL];    /* voltage/current step */
    double TRCVvSave[TRCVNESTLEVEL];    /* voltage of this source BEFORE 
                                         * analysis-to restore when done */
    int TRCVgSave[TRCVNESTLEVEL];    /* dcGiven flag; as with vSave */
    IFuid TRCVvName[TRCVNESTLEVEL];     /* source being varied */
    GENinstance *TRCVvElt[TRCVNESTLEVEL];   /* pointer to source */
    int TRCVvType[TRCVNESTLEVEL];   /* type of element being varied */
    int TRCVset[TRCVNESTLEVEL];     /* flag to indicate this nest level used */
    int TRCVnestLevel;      /* number of levels of nesting called for */
    int TRCVnestState;      /* iteration state during pause */
    int TRCVstepCount[TRCVNESTLEVEL]; /* count of accepted steps so far —
                                       * used by PARAM_CODE sweeps to compute
                                       * cur_val = start + N*step instead of
                                       * accumulating += step, which would
                                       * miss the exact endpoint by ~340 ULP
                                       * after 340 increments and break
                                       * `.measure ... when X=<endpoint>`. */
} TRCV;

enum {
    DCT_START1 = 1,
    DCT_STOP1,
    DCT_STEP1,
    DCT_NAME1,
    DCT_TYPE1,
    DCT_START2,
    DCT_STOP2,
    DCT_STEP2,
    DCT_NAME2,
    DCT_TYPE2,
};

#endif
