/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
Review: 2012-10 Francesco Lannutti
**********/

#ifndef ngspice_PSSDEFS_H
#define ngspice_PSSDEFS_H

#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
    /*
     * PSSdefs.h - defs for pss analyses
     */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
    double PSSguessedFreq;
    CKTnode *PSSoscNode;
    double PSSstabTime;
    long PSSmode;
    long int PSSpoints;
    int PSSharms;
    runDesc *PSSplot_td;
    runDesc *PSSplot_fd;
    int sc_iter;
    double steady_coeff;

    /* Enhancement-119: the converged periodic operating point, retained past the
     * analysis as the substrate the periodic small-signal analyses (PAC / pnoise
     * / PXF) linearize around. PSS already samples the node voltages over one
     * period for its DFT but frees them; here they -- and the device states,
     * which the reactive Jacobian C(t) needs -- are kept on the job. Row-major
     * per sample: [unknown + sample*PSSopMsize] and [state + sample*PSSopNumStates]. */
    long   PSSopPoints;      /* number of time samples over one period */
    int    PSSopMsize;       /* matrix size (nodes + branch currents) */
    int    PSSopNumStates;   /* CKTnumStates captured per sample */
    double PSSopFreq;        /* converged fundamental frequency (Hz) */
    double *PSSopTimes;      /* [PSSopPoints] sample times across the period */
    double *PSSopVoltages;   /* [PSSopMsize * PSSopPoints] node voltages per sample */
    double *PSSopStates;     /* [PSSopNumStates * PSSopPoints] device states per sample */

    /* Enhancement-122: PAC (.pac) frequency sweep. The .pac card runs PSS, then
     * sweeps a small-signal input frequency and, at each point, solves the
     * harmonic conversion matrix (E-121) built from the retained operating point,
     * emitting the sideband-0 node responses as a complex plot vs frequency. */
    int    PSSdoPAC;         /* 1 if this job also runs a PAC sweep */
    double PACfStart;        /* input-frequency sweep start (Hz) */
    double PACfStop;         /* input-frequency sweep stop (Hz) */
    int    PACpoints;        /* points per decade/octave, or total for linear */
    int    PACstepType;      /* 0 = LIN, 1 = DEC, 2 = OCT */
    int    PACmaxSideband;   /* number of output conversion sidebands each side (0 = sb0 only) */

    /* Enhancement-124: periodic noise (.pnoise). Runs PSS, then folds each device's
     * noise through the conversion-matrix adjoint over all sidebands to get the
     * output noise spectrum. Reuses the PAC sweep fields for the frequency sweep. */
    int      PSSdoPnoise;    /* 1 if this job runs a pnoise sweep */
    CKTnode *PnOutNode;      /* pnoise output node (reference = ground) */
    IFuid    PnInSrc;        /* input source name, for the input-referred spectrum */
    int      PSSpnCyclo;     /* Enhancement-126: 1 = cyclostationary noise (per-sample
                              * bias + time-domain transfer, averaged over the period);
                              * 0 = stationary (noise PSD at one operating point) */

    /* Enhancement-125: periodic transfer function (.pxf). The ADJOINT counterpart of
     * PAC: solve Hᵀ Ψ = e_{out,0} and dot Ψ with the netlist AC-source pattern to get
     * the transfer from the input to a fixed output at each sideband. Reuses the PAC
     * sweep fields + PACmaxSideband. */
    int      PSSdoPXF;       /* 1 if this job runs a PXF sweep */
    CKTnode *PxOutNode;      /* PXF output node (reference = ground) */

    /* Enhancement-132: periodic S-parameters (.psp). Runs PSS, then for each RF
     * port injects a unit current in sideband 0 through the conversion matrix and
     * reads the per-sideband port waves, building the periodic scattering matrix
     * S^(k) = B^(k)*A^-1 vs input frequency. Reuses the PAC sweep fields
     * (PACfStart/Stop/points/stepType/maxSideband). For a time-invariant circuit
     * the sideband-0 result reduces to the ordinary .sp S-matrix. */
    int      PSSdoPSP;       /* 1 if this job runs a periodic S-parameter sweep */
} PSSan;

enum {
    GUESSED_FREQ = 1,
    STAB_TIME,
    OSC_NODE,
    PSS_POINTS,
    PSS_HARMS,
    PSS_UIC,
    SC_ITER,
    STEADY_COEFF,
    PAC_DOPAC,
    PAC_FSTART,
    PAC_FSTOP,
    PAC_POINTS,
    PAC_STEPTYPE,
    PAC_MAXSB,
    PNOISE_DO,
    PNOISE_OUT,
    PNOISE_INSRC,
    PXF_DO,
    PXF_OUT,
    PNOISE_CYCLO,
    PSP_DOPSP,
};

#endif
