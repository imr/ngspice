/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /*
     *  CKTsetOpt(ckt,opt,value)
     *  set the specified 'opt' to have value 'value' in the
     *  given circuit 'ckt'.
     */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/optdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"

#include "analysis.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
/* gtri - begin - wbk - add includes */
#include "ngspice/mif.h"
/* gtri - end - wbk - add includes */
#endif

/* ARGSUSED */
int
CKTsetOpt(CKTcircuit *ckt, JOB *anal, int opt, IFvalue *val)
{
    TSKtask *task = (TSKtask *)anal;

    NG_IGNORE(ckt);

    switch(opt) {

    case OPT_NOOPITER:
        task->TSKnoOpIter = (val->iValue != 0);
        break;
    case OPT_GMIN:
        task->TSKgmin = val->rValue;
        break;
     case OPT_GSHUNT:
        task->TSKgshunt = val->rValue;
        break;
    case OPT_RELTOL:
        task->TSKreltol = val->rValue;
        break;
    case OPT_ABSTOL:
        task->TSKabstol = val->rValue;
        break;
    case OPT_VNTOL:
        task->TSKvoltTol = val->rValue;
        break;
    case OPT_TRTOL:
        task->TSKtrtol = val->rValue;
        break;
    case OPT_CHGTOL:
        task->TSKchgtol = val->rValue;
        break;
    case OPT_PIVTOL:
        task->TSKpivotAbsTol = val->rValue;
        break;
    case OPT_PIVREL:
        task->TSKpivotRelTol = val->rValue;
        break;
    case OPT_TNOM:
        task->TSKnomTemp = val->rValue + CONSTCtoK; /* Centegrade to Kelvin */
        break;
    case OPT_TEMP:
        task->TSKtemp = val->rValue + CONSTCtoK; /* Centegrade to Kelvin */
        break;
    case OPT_ITL1:
        task->TSKdcMaxIter = val->iValue;
        break;
    case OPT_ITL2:
        task->TSKdcTrcvMaxIter = val->iValue;
        break;
    case OPT_ITL3:
        break;
    case OPT_ITL4:
        task->TSKtranMaxIter = val->iValue;
        break;
    case OPT_ITL5:
        break;
    case OPT_SRCSTEPS:
        task->TSKnumSrcSteps = val->iValue;
        break;
    case OPT_GMINSTEPS:
        task->TSKnumGminSteps = val->iValue;
        break;
    case OPT_GMINFACT:
        task->TSKgminFactor = val->rValue;
        break;
    case OPT_DEFM:
        task->TSKdefaultMosM = val->rValue;
        break;
    case OPT_DEFL:
        task->TSKdefaultMosL = val->rValue;
        break;
    case OPT_DEFW:
        task->TSKdefaultMosW = val->rValue;
        break;
    case OPT_DEFAD:
        task->TSKdefaultMosAD = val->rValue;
        break;
    case OPT_DEFAS:
        task->TSKdefaultMosAD = val->rValue;
        break;
    case OPT_BYPASS:
        task->TSKbypass = val->iValue;
        break;
    case OPT_INDVERBOSITY:
        task->TSKindverbosity = val->iValue;
        break;
    case OPT_XMU:
        task->TSKxmu = val->rValue;
        break;
    case OPT_MAXORD:
        task->TSKmaxOrder = val->iValue;
        /* Check options method and maxorder for consistency */
        if (task->TSKmaxOrder < 1) {
            task->TSKmaxOrder = 1;
            fprintf(stderr,"\nWarning -- Option maxord < 1 not allowed in ngspice\nSet to 1\n\n");
        }
        else if (task->TSKmaxOrder > 6) {
            task->TSKmaxOrder = 6;
            fprintf(stderr,"\nWarning -- Option maxord > 6 not allowed in ngspice\nSet to 6\n\n");
        }
        break;
    case OPT_OLDLIMIT:
        task->TSKfixLimit = (val->iValue != 0);
        break;
    case OPT_MINBREAK:
        task->TSKminBreak = val->rValue;
        break;
    case OPT_METHOD:
        if(strncmp(val->sValue,"trap", 4)==0)
            task->TSKintegrateMethod=TRAPEZOIDAL;
        else if (strcmp(val->sValue,"gear")==0)
            task->TSKintegrateMethod=GEAR;
        else return(E_METHOD);
        break;
    case OPT_TRYTOCOMPACT:
        task->TSKtryToCompact = (val->iValue != 0);
        break;
    case OPT_BADMOS3:
        task->TSKbadMos3 = (val->iValue != 0);
        break;
    case OPT_KEEPOPINFO:
        task->TSKkeepOpInfo = (val->iValue != 0);
        break;
    case OPT_COPYNODESETS:
        task->TSKcopyNodesets = (val->iValue != 0);
        break;
    case OPT_NODEDAMPING:
        task->TSKnodeDamping = (val->iValue != 0);
        break;
    case OPT_ABSDV:
        task->TSKabsDv = val->rValue;
        break;
    case OPT_RELDV:
        task->TSKrelDv = val->rValue;
        break;
    case OPT_NOOPAC:
        task->TSKnoopac = (val->iValue != 0);
        break;
    case OPT_EPSMIN:
        task->TSKepsmin = val->rValue;
        break;
    case OPT_CSHUNT:
        task->TSKcshunt = val->rValue;
        break;
/* gtri - begin - wbk - add new options */
#ifdef XSPICE
    case OPT_EVT_MAX_OP_ALTER:
        ckt->evt->limits.max_op_alternations = val->iValue;
        break;

    case OPT_EVT_MAX_EVT_PASSES:
        ckt->evt->limits.max_event_passes = val->iValue;
        break;

    case OPT_ENH_NOOPALTER:
        ckt->evt->options.op_alternate = MIF_FALSE;
        break;

    case OPT_ENH_RAMPTIME:
        ckt->enh->ramp.ramptime = val->rValue;
        break;

    case OPT_ENH_CONV_LIMIT:
        ckt->enh->conv_limit.enabled = MIF_TRUE;
        break;

    case OPT_ENH_CONV_STEP:
        ckt->enh->conv_limit.step = val->rValue;
        ckt->enh->conv_limit.enabled = MIF_TRUE;
        break;

    case OPT_ENH_CONV_ABS_STEP:
        ckt->enh->conv_limit.abs_step = val->rValue;
        ckt->enh->conv_limit.enabled = MIF_TRUE;
        break;

    case OPT_MIF_AUTO_PARTIAL:
        g_mif_info.auto_partial.global = MIF_TRUE;
        break;

    case OPT_ENH_RSHUNT:
        if(val->rValue > 1.0e-30) {
          ckt->enh->rshunt_data.enabled = MIF_TRUE;
          ckt->enh->rshunt_data.gshunt = 1.0 / val->rValue;
        }
        else {
          printf("WARNING - Rshunt option too small.  Ignored.\n");
        }
        break;
#endif
/* gtri - end - wbk - add new options */
    default:
        return(-1);
    }
    return(0);
}
static IFparm OPTtbl[] = {
#ifdef XSPICE
/* gtri - begin - wbk - add new options */
 { "maxopalter", OPT_EVT_MAX_OP_ALTER, IF_SET|IF_INTEGER, "Maximum analog/event alternations in DCOP" },
 { "maxevtiter", OPT_EVT_MAX_EVT_PASSES, IF_SET|IF_INTEGER, "Maximum event iterations at analysis point" },
 { "noopalter", OPT_ENH_NOOPALTER, IF_SET|IF_FLAG, "Do not do analog/event alternation in DCOP" },
 { "ramptime", OPT_ENH_RAMPTIME, IF_SET|IF_REAL, "Transient analysis supply ramping time" },
 { "convlimit", OPT_ENH_CONV_LIMIT, IF_SET|IF_FLAG, "Enable convergence assistance on code models" },
 { "convstep", OPT_ENH_CONV_STEP, IF_SET|IF_REAL, "Fractional step allowed by code model inputs between iterations" },
 { "convabsstep", OPT_ENH_CONV_ABS_STEP, IF_SET|IF_REAL, "Absolute step allowed by code model inputs between iterations" },
 { "autopartial", OPT_MIF_AUTO_PARTIAL, IF_SET|IF_FLAG, "Use auto-partial computation for all models" },
 { "rshunt", OPT_ENH_RSHUNT, IF_SET|IF_REAL, "Shunt resistance from analog nodes to ground" },
/* gtri - end   - wbk - add new options */
#endif
 { "cshunt", OPT_CSHUNT, IF_SET|IF_REAL, "Shunt capacitor from analog nodes to ground" },
 { "noopiter", OPT_NOOPITER,IF_SET|IF_FLAG,"Go directly to gmin stepping" },
 { "gmin", OPT_GMIN,IF_SET|IF_REAL,"Minimum conductance" },
 { "gshunt", OPT_GSHUNT,IF_SET|IF_REAL,"Shunt conductance" },
 { "reltol", OPT_RELTOL,IF_SET|IF_REAL ,"Relative error tolerence"},
 { "abstol", OPT_ABSTOL,IF_SET|IF_REAL,"Absolute error tolerence" },
 { "vntol", OPT_VNTOL,IF_SET|IF_REAL,"Voltage error tolerence" },
 { "trtol", OPT_TRTOL,IF_SET|IF_REAL,"Truncation error overestimation factor" },
 { "chgtol", OPT_CHGTOL,IF_SET|IF_REAL, "Charge error tolerence" },
 { "pivtol", OPT_PIVTOL,IF_SET|IF_REAL, "Minimum acceptable pivot" },
 { "pivrel", OPT_PIVREL,IF_SET|IF_REAL, "Minimum acceptable ratio of pivot" },
 { "tnom", OPT_TNOM,IF_SET|IF_ASK|IF_REAL, "Nominal temperature" },
 { "temp", OPT_TEMP,IF_SET|IF_ASK|IF_REAL, "Operating temperature" },
 { "itl1", OPT_ITL1,IF_SET|IF_INTEGER,"DC iteration limit" },
 { "itl2", OPT_ITL2,IF_SET|IF_INTEGER,"DC transfer curve iteration limit" },
 { "itl3", OPT_ITL3, IF_INTEGER,"Lower transient iteration limit"},
 { "itl4", OPT_ITL4,IF_SET|IF_INTEGER,"Upper transient iteration limit" },
 { "itl5", OPT_ITL5, IF_INTEGER,"Total transient iteration limit"},
 { "itl6", OPT_SRCSTEPS, IF_SET|IF_INTEGER,"number of source steps"},
 { "srcsteps", OPT_SRCSTEPS, IF_SET|IF_INTEGER,"number of source steps"},
 { "gminsteps", OPT_GMINSTEPS, IF_SET|IF_INTEGER,"number of Gmin steps"},
 { "gminfactor", OPT_GMINFACT, IF_SET|IF_REAL,"factor per Gmin step"},
 { "acct", 0, IF_FLAG ,"Print accounting"},
 { "list", 0, IF_FLAG, "Print a listing" },
 { "nomod", 0, IF_FLAG, "Don't print a model summary" },
 { "nopage", 0, IF_FLAG, "Don't insert page breaks" },
 { "node", 0, IF_FLAG,"Print a node connection summary" },
 { "opts", 0, IF_FLAG, "Print a list of the options" },
 { "oldlimit", OPT_OLDLIMIT, IF_SET|IF_FLAG, "use SPICE2 MOSfet limiting" },
 { "numdgt", 0, IF_INTEGER, "Set number of digits printed"},
 { "cptime", 0, IF_REAL, "Total cpu time in seconds" },
 { "limtim", 0, IF_INTEGER, "Time to reserve for output" },
 { "limpts", 0,IF_INTEGER,"Maximum points per analysis"},
 { "lvlcod", 0, IF_INTEGER,"Generate machine code" },
 { "lvltim", 0, IF_INTEGER,"Type of timestep control" },
 { "method", OPT_METHOD, IF_SET|IF_STRING,"Integration method" },
 { "maxord", OPT_MAXORD, IF_SET|IF_INTEGER,"Maximum integration order" },
 { "indverbosity", OPT_INDVERBOSITY, IF_SET|IF_INTEGER,"Control Inductive Systems Check (coupling)" },
 { "xmu", OPT_XMU, IF_SET|IF_REAL,"Coefficient for trapezoidal method" },
 { "defm", OPT_DEFM,IF_SET|IF_REAL,"Default MOSfet Multiplier" },
 { "defl", OPT_DEFL,IF_SET|IF_REAL,"Default MOSfet length" },
 { "defw", OPT_DEFW,IF_SET|IF_REAL,"Default MOSfet width" },
 { "minbreak", OPT_MINBREAK,IF_SET|IF_REAL,"Minimum time between breakpoints" },
 { "defad", OPT_DEFAD,IF_SET|IF_REAL,"Default MOSfet area of drain" },
 { "defas", OPT_DEFAS,IF_SET|IF_REAL,"Default MOSfet area of source" },
 { "bypass",OPT_BYPASS,IF_SET|IF_INTEGER,"Allow bypass of unchanging elements"},
 { "totiter", OPT_ITERS, IF_ASK|IF_INTEGER,"Total iterations" },
 { "traniter", OPT_TRANIT, IF_ASK|IF_INTEGER ,"Transient iterations"},
 { "equations", OPT_EQNS, IF_ASK|IF_INTEGER,"Circuit Equations" },
 { "originalnz", OPT_ORIGNZ, IF_ASK|IF_INTEGER,"Circuit original non-zeroes" },
 { "fillinnz", OPT_FILLNZ, IF_ASK|IF_INTEGER,"Circuit fill-in non-zeroes" },
 { "totalnz", OPT_TOTALNZ, IF_ASK|IF_INTEGER,"Circuit total non-zeroes" },
 { "tranpoints", OPT_TRANPTS, IF_ASK|IF_INTEGER,"Transient timepoints" },
 { "accept", OPT_TRANACCPT, IF_ASK|IF_INTEGER,"Accepted timepoints" },
 { "rejected", OPT_TRANRJCT, IF_ASK|IF_INTEGER,"Rejected timepoints" },
 { "time", OPT_TOTANALTIME, IF_ASK|IF_REAL,"Total analysis time (seconds)" },
 { "loadtime", OPT_LOADTIME, IF_ASK|IF_REAL,"Matrix load time" },
 { "synctime", OPT_SYNCTIME, IF_ASK|IF_REAL,"Matrix synchronize time" },
 { "reordertime", OPT_REORDTIME, IF_ASK|IF_REAL,"Matrix reorder time" },
 { "factortime", OPT_DECOMP, IF_ASK|IF_REAL,"Matrix factor time" },
 { "solvetime", OPT_SOLVE, IF_ASK|IF_REAL,"Matrix solve time" },
 { "trantime", OPT_TRANTIME, IF_ASK|IF_REAL,"Transient analysis time" },
 { "tranloadtime", OPT_TRANLOAD, IF_ASK|IF_REAL,"Transient load time" },
 { "transynctime", OPT_TRANSYNC, IF_ASK|IF_REAL,"Transient sync time" },
 { "tranfactortime", OPT_TRANDECOMP,IF_ASK|IF_REAL,"Transient factor time" },
 { "transolvetime", OPT_TRANSOLVE, IF_ASK|IF_REAL,"Transient solve time" },
 { "trantrunctime", OPT_TRANTRUNC, IF_ASK|IF_REAL,"Transient trunc time" },
 { "trancuriters", OPT_TRANCURITER, IF_ASK|IF_INTEGER,
        "Transient iterations for the last time point" },
 { "actime", OPT_ACTIME, IF_ASK|IF_REAL,"AC analysis time" },
 { "acloadtime", OPT_ACLOAD, IF_ASK|IF_REAL,"AC load time" },
 { "acsynctime", OPT_ACSYNC, IF_ASK|IF_REAL,"AC sync time" },
 { "acfactortime", OPT_ACDECOMP,IF_ASK|IF_REAL,"AC factor time" },
 { "acsolvetime", OPT_ACSOLVE, IF_ASK|IF_REAL,"AC solve time" },
 { "trytocompact", OPT_TRYTOCOMPACT, IF_SET|IF_FLAG,
        "Try compaction for LTRA lines" },
 { "badmos3", OPT_BADMOS3, IF_SET|IF_FLAG,
        "use old mos3 model (discontinuous with respect to kappa)" },
 { "keepopinfo", OPT_KEEPOPINFO, IF_SET|IF_FLAG,
        "Record operating point for each small-signal analysis" },
 { "copynodesets", OPT_COPYNODESETS, IF_SET|IF_FLAG,
        "Copy nodesets from device terminals to internal nodes" },
 { "nodedamping", OPT_NODEDAMPING, IF_SET|IF_FLAG,
        "Limit iteration to iteration node voltage change" },
 { "absdv", OPT_ABSDV, IF_SET|IF_REAL,
        "Maximum absolute iter-iter node voltage change" },
 { "reldv", OPT_RELDV, IF_SET|IF_REAL,
        "Maximum relative iter-iter node voltage change" },
 { "noopac", OPT_NOOPAC, IF_SET|IF_FLAG,
        "No op calculation in ac if circuit is linear" },
 { "epsmin", OPT_EPSMIN, IF_SET|IF_REAL,
        "Minimum value for log" }
};

int OPTcount = NUMELEMS(OPTtbl);

SPICEanalysis OPTinfo = {
    {
        "options",
        "Task option selection",
        NUMELEMS(OPTtbl),
        OPTtbl
    },
    0, /* no size associated with options */
    NODOMAIN,
    0,
    CKTsetOpt,
    CKTacct,
    NULL,
    NULL
};
