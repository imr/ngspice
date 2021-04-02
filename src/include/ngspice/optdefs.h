/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef ngspice_OPTDEFS_H
#define ngspice_OPTDEFS_H

    /* structure used to describe the statistics to be collected */

typedef struct sSTATdevList {
    struct sSTATdevList *STATnextDev;
    int modNum;
    int instNum;
} STATdevList;

typedef struct {

    int STATnumIter;    /* number of total iterations performed */
    int STATtranIter;   /* number of iterations for transient analysis */
    int STAToldIter;    /* number of iterations at the end of the last point */
                        /* used to compute iterations per point */

    int STATtimePts;    /* total number of timepoints */
    int STATaccepted;   /* number of timepoints accepted */
    int STATrejected;   /* number of timepoints rejected */

    int STATtotalDev;   /* PN: number of total devices in the netlist */

    double STATtotAnalTime;     /* total time for all analysis */
    double STATloadTime;        /* total time spent in device loading */
    double STATdecompTime;      /* total time spent in LU decomposition */
    double STATsolveTime;       /* total time spent in F-B subst. */
    double STATreorderTime;     /* total time spent reordering */
    double STATsyncTime;        /* total time spent sync'ing after load */
    double STATtranTime;        /* transient analysis time */
    double STATtranDecompTime;  /* time spent in transient LU decomposition */
    double STATtranSolveTime;   /* time spent in transient F-B subst. */
    double STATtranLoadTime;    /* time spent in transient device loading */
    double STATtranTruncTime;   /* time spent calculating LTE and new step */
    double STATtranSyncTime;    /* time spent in transient sync'ing */
    double STATacTime;          /* AC analysis time */
    double STATacDecompTime;    /* time spent in AC LU decomposition */
    double STATacSolveTime;     /* time spent in AC F-B subst. */
    double STATacLoadTime;      /* time spent in AC device loading */
    double STATacSyncTime;      /* time spent in transient sync'ing */
    STATdevList *STATdevNum;    /* PN: Number of instances and models for each device */
} STATistics;

enum {
    OPT_GMIN = 1,
    OPT_RELTOL,
    OPT_ABSTOL,
    OPT_VNTOL,
    OPT_TRTOL,
    OPT_CHGTOL,
    OPT_PIVTOL,
    OPT_PIVREL,
    OPT_TNOM,
    OPT_ITL1,
    OPT_ITL2,
    OPT_ITL3,
    OPT_ITL4,
    OPT_ITL5,
    OPT_DEFL,
    OPT_DEFW,
    OPT_DEFAD,
    OPT_DEFAS,
    OPT_BYPASS,
    OPT_MAXORD,
    OPT_ITERS,
    OPT_TRANIT,
    OPT_TRANPTS,
    OPT_TRANACCPT,
    OPT_TRANRJCT,
    OPT_TOTANALTIME,
    OPT_TRANTIME,
    OPT_LOADTIME,
    OPT_DECOMP,
    OPT_SOLVE,
    OPT_TRANDECOMP,
    OPT_TRANSOLVE,
    OPT_TEMP,
    OPT_OLDLIMIT,
    OPT_TRANCURITER,
    OPT_SRCSTEPS,
    OPT_GMINSTEPS,
    OPT_MINBREAK,
    OPT_NOOPITER,
    OPT_EQNS,
    OPT_REORDTIME,
    OPT_METHOD,
    OPT_TRYTOCOMPACT,
    OPT_BADMOS3,
    OPT_KEEPOPINFO,
    OPT_TRANLOAD,
    OPT_TRANTRUNC,
    OPT_ACTIME,
    OPT_ACLOAD,
    OPT_ACDECOMP,
    OPT_ACSOLVE,
    OPT_ORIGNZ,
    OPT_FILLNZ,
    OPT_TOTALNZ,
};

enum {
    OPT_SYNCTIME = 58,
    OPT_TRANSYNC,
    OPT_ACSYNC,
    OPT_GSHUNT,
    OPT_DEFM,
    OPT_GMINFACT,
    OPT_COPYNODESETS,
    OPT_NODEDAMPING,
    OPT_ABSDV,
    OPT_RELDV,
    OPT_NOOPAC,
    OPT_XMU,
    OPT_INDVERBOSITY,
    OPT_EPSMIN,
    OPT_CSHUNT,
};

#ifdef XSPICE
/* gtri - begin - wbk - add new options */
enum {
    OPT_ENH_NOOPALTER = 100,
    OPT_ENH_RAMPTIME,
    OPT_EVT_MAX_EVT_PASSES,
    OPT_EVT_MAX_OP_ALTER,
    OPT_ENH_CONV_LIMIT,
    OPT_ENH_CONV_ABS_STEP,
    OPT_ENH_CONV_STEP,
    OPT_MIF_AUTO_PARTIAL,
    OPT_ENH_RSHUNT,
};

/* gtri - end   - wbk - add new options */
#endif

#define OPT_TOTALDEV 200 /* Total devices in the netlist */

#endif
