/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef OPT
#define OPT

    /* structure used to describe the statistics to be collected */

typedef struct {

    int STATnumIter;    /* number of total iterations performed */
    int STATtranIter;   /* number of iterations for transient analysis */
    int STAToldIter;    /* number of iterations at the end of the last point */
                        /* used to compute iterations per point */

    int STATtimePts;    /* total number of timepoints */
    int STATaccepted;   /* number of timepoints accepted */
    int STATrejected;   /* number of timepoints rejected */

    double STATtotAnalTime;     /* total time for all analysis */
    double STATloadTime;        /* total time spent in device loading */
    double STATdecompTime;      /* total time spent in LU decomposition */
    double STATsolveTime;       /* total time spent in F-B subst. */
    double STATreorderTime;     /* total time spent reordering */
    double STATcombineTime;     /* total time spent combining */
    double STATsyncTime;        /* total time spent sync'ing after load */
    double STATtranTime;        /* transient analysis time */
    double STATtranDecompTime;  /* time spent in transient LU decomposition */
    double STATtranSolveTime;   /* time spent in transient F-B subst. */
    double STATtranLoadTime;    /* time spent in transient device loading */
    double STATtranTruncTime;   /* time spent calculating LTE and new step */
    double STATtranCombTime;    /* time spent in transient combining */
    double STATtranSyncTime;    /* time spent in transient sync'ing */
    double STATacTime;          /* AC analysis time */
    double STATacDecompTime;    /* time spent in AC LU decomposition */
    double STATacSolveTime;     /* time spent in AC F-B subst. */
    double STATacLoadTime;      /* time spent in AC device loading */
    double STATacCombTime;      /* time spent in AC combining */
    double STATacSyncTime;      /* time spent in transient sync'ing */

} STATistics;

#define OPT_GMIN 1
#define OPT_RELTOL 2
#define OPT_ABSTOL 3
#define OPT_VNTOL 4
#define OPT_TRTOL 5
#define OPT_CHGTOL 6
#define OPT_PIVTOL 7
#define OPT_PIVREL 8
#define OPT_TNOM 9
#define OPT_ITL1 10
#define OPT_ITL2 11
#define OPT_ITL3 12
#define OPT_ITL4 13
#define OPT_ITL5 14
#define OPT_DEFL 15
#define OPT_DEFW 16
#define OPT_DEFAD 17
#define OPT_DEFAS 18
#define OPT_BYPASS 19
#define OPT_MAXORD 20

#define OPT_ITERS 21
#define OPT_TRANIT 22
#define OPT_TRANPTS 23
#define OPT_TRANACCPT 24
#define OPT_TRANRJCT 25
#define OPT_TOTANALTIME 26
#define OPT_TRANTIME 27
#define OPT_LOADTIME 28
#define OPT_DECOMP 29
#define OPT_SOLVE 30
#define OPT_TRANDECOMP 31
#define OPT_TRANSOLVE 32
#define OPT_TEMP 33
#define OPT_OLDLIMIT 34
#define OPT_TRANCURITER 35
#define OPT_SRCSTEPS 36
#define OPT_GMINSTEPS 37
#define OPT_MINBREAK 38
#define OPT_NOOPITER 39
#define OPT_EQNS 40
#define OPT_REORDTIME 41
#define OPT_METHOD 42
#define OPT_TRYTOCOMPACT 43
#define OPT_BADMOS3 44
#define OPT_KEEPOPINFO 45
#define OPT_TRANLOAD 46
#define OPT_TRANTRUNC 47
#define OPT_ACTIME 48
#define OPT_ACLOAD 49
#define OPT_ACDECOMP 50
#define OPT_ACSOLVE 51
#define OPT_ORIGNZ 52
#define OPT_FILLNZ 53
#define OPT_TOTALNZ 54
#define OPT_COMBTIME 55
#define OPT_TRANCOMB 56
#define OPT_ACCOMB 57
#define OPT_SYNCTIME 58
#define OPT_TRANSYNC 59
#define OPT_ACSYNC 60

/* AlansFixes: It is not possible to use the same numbers of the
   original Alan code. The following options are all AlansFixes */
    
#define OPT_GSHUNT       61  /* Original: 48 */
#define OPT_DEFM         62  /* Original: 46 */
#define OPT_GMINFACT     63  /* Original: 47 */
#define OPT_COPYNODESETS 64  /* Original: 49 (NodesetFix) */
#define OPT_NODEDAMPING  65  /* Original: 50 (Node_Damping) */
#define OPT_ABSDV        66  /* Original: 51 (Node_Damping) */ 
#define OPT_RELDV        67  /* Original: 52 (Node_Damping) */

#ifdef XSPICE
/* gtri - begin - wbk - add new options */
#define OPT_ENH_NOOPALTER           100
#define OPT_ENH_RAMPTIME            101
#define OPT_EVT_MAX_EVT_PASSES      102
#define OPT_EVT_MAX_OP_ALTER        103
#define OPT_ENH_CONV_LIMIT          104
#define OPT_ENH_CONV_ABS_STEP       105
#define OPT_ENH_CONV_STEP           106
#define OPT_MIF_AUTO_PARTIAL        107
#define OPT_ENH_RSHUNT              108
/* gtri - end   - wbk - add new options */
#endif

#endif /*OPT*/
