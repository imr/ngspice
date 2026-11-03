/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/pssdefs.h"

#include "analysis.h"

/* ARGSUSED */
int
PSSsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    PSSan *job = (PSSan *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case GUESSED_FREQ:
        job->PSSguessedFreq = value->rValue;
        break;
    case OSC_NODE:
        job->PSSoscNode = value->nValue;
        break;
    case STAB_TIME:
        job->PSSstabTime = value->rValue;
        break;
    case PSS_POINTS:
        job->PSSpoints = value->iValue;
        break;
    case PSS_HARMS:
        job->PSSharms = value->iValue;
        break;
    case PSS_UIC:
        if(value->iValue) {
            job->PSSmode |= MODEUIC;
        }
        break;
    case SC_ITER:
        job->sc_iter = value->iValue;
        break;
    case STEADY_COEFF:
        job->steady_coeff = value->rValue;
        break;

    /* Enhancement-122: PAC sweep parameters (.pac card) */
    case PAC_DOPAC:
        job->PSSdoPAC = value->iValue;
        break;
    case PAC_FSTART:
        job->PACfStart = value->rValue;
        break;
    case PAC_FSTOP:
        job->PACfStop = value->rValue;
        break;
    case PAC_POINTS:
        job->PACpoints = value->iValue;
        break;
    case PAC_STEPTYPE:
        job->PACstepType = value->iValue;
        break;
    case PAC_MAXSB:
        job->PACmaxSideband = value->iValue;
        break;

    /* Enhancement-124: pnoise parameters (.pnoise card) */
    case PNOISE_DO:
        job->PSSdoPnoise = value->iValue;
        break;
    case PNOISE_OUT:
        job->PnOutNode = value->nValue;
        break;
    case PNOISE_INSRC:
        job->PnInSrc = value->uValue;
        break;

    /* Enhancement-125: pxf parameters (.pxf card) */
    case PXF_DO:
        job->PSSdoPXF = value->iValue;
        break;
    case PXF_OUT:
        job->PxOutNode = value->nValue;
        break;

    /* Enhancement-126: cyclostationary-noise flag */
    case PNOISE_CYCLO:
        job->PSSpnCyclo = value->iValue;
        break;

    /* Enhancement-132: periodic S-parameters flag */
    case PSP_DOPSP:
        job->PSSdoPSP = value->iValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm PSSparms[] = {
    { "fguess",    GUESSED_FREQ,	IF_SET|IF_REAL, 	"guessed frequency" },
    { "oscnode",   OSC_NODE,		IF_SET|IF_STRING,	"oscillation node" },
    { "stabtime",  STAB_TIME,		IF_SET|IF_REAL,		"stabilization time" },
    { "points",    PSS_POINTS,		IF_SET|IF_INTEGER, 	"pick equispaced number of time points in PSS" },
    { "harmonics", PSS_HARMS,		IF_SET|IF_INTEGER, 	"consider only given number of harmonics in PSS from DC" },
    { "uic",       PSS_UIC,		IF_SET|IF_INTEGER, 	"use initial conditions (1 true - 0 false)" },
    { "sc_iter",   SC_ITER,		IF_SET|IF_INTEGER, 	"maxmimum number of shooting cycle iterations" },
    { "steady_coeff",   STEADY_COEFF,	IF_SET|IF_INTEGER, 	"set steady coefficient for convergence test" },
    { "pac",        PAC_DOPAC,		IF_SET|IF_INTEGER,	"run a PAC input-frequency sweep after PSS" },
    { "pac_fstart", PAC_FSTART,		IF_SET|IF_REAL,		"PAC input sweep start frequency" },
    { "pac_fstop",  PAC_FSTOP,		IF_SET|IF_REAL,		"PAC input sweep stop frequency" },
    { "pac_points", PAC_POINTS,		IF_SET|IF_INTEGER,	"PAC points per decade/octave (or total for linear)" },
    { "pac_step",   PAC_STEPTYPE,	IF_SET|IF_INTEGER,	"PAC sweep step type (0 lin, 1 dec, 2 oct)" },
    { "pac_maxsb",  PAC_MAXSB,		IF_SET|IF_INTEGER,	"PAC output conversion sidebands each side (0 = sideband 0 only)" },
    { "pnoise",       PNOISE_DO,	IF_SET|IF_INTEGER,	"run a periodic-noise sweep after PSS" },
    { "pnoise_out",   PNOISE_OUT,	IF_SET|IF_STRING,	"pnoise output node" },
    { "pnoise_insrc", PNOISE_INSRC,	IF_SET|IF_STRING,	"pnoise input source (for the input-referred spectrum)" },
    { "pxf",          PXF_DO,		IF_SET|IF_INTEGER,	"run a periodic transfer-function sweep after PSS" },
    { "pxf_out",      PXF_OUT,		IF_SET|IF_STRING,	"pxf output node" },
    { "pnoise_cyclo", PNOISE_CYCLO,	IF_SET|IF_INTEGER,	"pnoise cyclostationary mode (per-sample bias, period average)" },
    { "psp",          PSP_DOPSP,	IF_SET|IF_INTEGER,	"run a periodic S-parameter sweep after PSS (needs RF ports)" }
};

SPICEanalysis PSSinfo  = {
    {
        "PSS",
        "Periodic Steady State analysis",

        sizeof(PSSparms)/sizeof(IFparm),
        PSSparms
    },
    sizeof(PSSan),
    TIMEDOMAIN,
    1,
    PSSsetParm,
    PSSaskQuest,
    PSSinit,
    DCpss
};
