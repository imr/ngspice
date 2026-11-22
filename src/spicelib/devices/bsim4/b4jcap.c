/**********
 * BSIM4 source/drain-bulk junction charge and capacitance
 *
 * Extracted from BSIM4load() (b4ld.c, BSIM4 4.8.3) without changing the
 * equations, so that alternative junction capacitance models (e.g. JUNCAP2)
 * can later be selected by a model switch.
 *
 * Licensed under Educational Community License, Version 2.0, like BSIM4.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"
#include "ngspice/const.h"

/*
 * One depletion component (bottom, STI sidewall or gate sidewall) in reverse
 * bias (vj < 0).  Adds its charge to *qj and its capacitance to *cj.
 *
 *   cz   zero-bias capacitance of this component [F] (geometry included)
 *   phi  built-in potential [V] (already temperature scaled)
 *   mj   grading coefficient
 */
static void
BSIM4jctDeplRev(double vj, double cz, double phi, double mj,
                double *qj, double *cj)
{
    double arg, sarg;

    if (cz > 0.0) {
        arg = 1.0 - vj / phi;
        if (mj == 0.5)
            sarg = 1.0 / sqrt(arg);
        else
            sarg = exp(-mj * log(arg));
        *qj += phi * cz * (1.0 - arg * sarg) / (1.0 - mj);
        *cj += cz * sarg;
    }
}

/*
 * Standard BSIM4 junction C-V model for one junction (source or drain).
 *
 * Inputs
 *   vj      junction voltage bulk - S/D [V], already multiplied by type
 *           (vbs_jct or vbd_jct in BSIM4load)
 *   cz*     zero-bias capacitances [F] of bottom, sidewall, gate sidewall
 *   phi*    built-in potentials [V] (PhiBS/PhiBSWS/PhiBSWGS resp. drain)
 *   mj*     grading coefficients
 * Outputs
 *   *qj     junction charge [C]
 *   *cj     junction capacitance dQ/dV [F]
 *
 * The function only depends on its arguments (no static data), so it is
 * safe to call from the OpenMP load path.
 */
void
BSIM4jctCapStd(double vj,
               double cz, double czsw, double czswg,
               double phi, double phisw, double phiswg,
               double mj, double mjsw, double mjswg,
               double *qj, double *cj)
{
    double T0, T1;

    *qj = 0.0;
    *cj = 0.0;

    if (vj == 0.0) {
        *cj = cz + czsw + czswg;
    }
    else if (vj < 0.0) {
        BSIM4jctDeplRev(vj, cz,    phi,    mj,    qj, cj);
        BSIM4jctDeplRev(vj, czsw,  phisw,  mjsw,  qj, cj);
        BSIM4jctDeplRev(vj, czswg, phiswg, mjswg, qj, cj);
    }
    else {
        /* forward bias: linear extrapolation of C(V) from V = 0 */
        T0 = cz + czsw + czswg;
        T1 = vj * (cz * mj / phi + czsw * mjsw / phisw + czswg * mjswg / phiswg);
        *qj = vj * (T0 + 0.5 * T1);
        *cj = T0 + T1;
    }
}


/*************************************************************************
 *
 * JUNCAP2 (juncap = 2), shared code in devices/juncap2.c
 *
 *************************************************************************/

/* default JUNCAP2 reference temperature [K] (TRJ not given), as in other
   simulators: TNOM if given in the model card, else 27 C; the option TNOM
   has no effect */
static double
BSIM4jc2trdef(BSIM4model *mod)
{
    return mod->BSIM4tnomGiven ? mod->BSIM4tnom : CONSTCtoK + 27.0;
}

int
BSIM4jc2mParam(int param, IFvalue *value, BSIM4model *mod)
{
    if (param == BSIM4_MOD_JUNCAP) {
        mod->BSIM4juncap = value->iValue;
        mod->BSIM4juncapGiven = TRUE;
        return OK;
    }
    if (param >= BSIM4_MOD_JC2BASE && param < BSIM4_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mParam(&mod->BSIM4jc2, param - BSIM4_MOD_JC2BASE, value);
    return E_BADPARM;
}

int
BSIM4jc2mAsk(BSIM4model *mod, int which, IFvalue *value)
{
    if (which == BSIM4_MOD_JUNCAP) {
        value->iValue = mod->BSIM4juncap;
        return OK;
    }
    if (which >= BSIM4_MOD_JC2BASE && which < BSIM4_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mAsk(&mod->BSIM4jc2, BSIM4jc2trdef(mod), which - BSIM4_MOD_JC2BASE, value);
    return E_BADPARM;
}

/* juncap selector and JUNCAP2 defaults, called from BSIM4setup() */
void
BSIM4jc2Setup(BSIM4model *mod)
{
    if (!mod->BSIM4juncapGiven) {
        mod->BSIM4juncap = 0;
    } else if (mod->BSIM4juncap == 1) {
        printf("Warning: juncap = 1 (JUNCAP1) is not implemented, juncap has been set to 0.\n");
        mod->BSIM4juncap = 0;
    } else if ((mod->BSIM4juncap != 0) && (mod->BSIM4juncap != 2)) {
        printf("Warning: juncap = %d not matched, juncap has been set to 0.\n", mod->BSIM4juncap);
        mod->BSIM4juncap = 0;
    }
    JC2setup(&mod->BSIM4jc2, mod->BSIM4modName, mod->BSIM4juncap == 2);
}

/* called from BSIM4temp() */
void
BSIM4jc2TempModel(BSIM4model *mod, double Temp)
{
    if (mod->BSIM4juncap == 2)
        JC2temp(&mod->BSIM4jc2, &mod->BSIM4jc2td, Temp, BSIM4jc2trdef(mod));
}

/* called from BSIM4temp() per instance, after the effective S/D geometry is
   known; Pseff/Pdeff without gate edge, LG = Weff_CJ * NF */
void
BSIM4jc2TempInst(BSIM4model *mod, BSIM4instance *here,
                 double Aseff, double Pseff, double Adeff, double Pdeff, double LG)
{
    if (mod->BSIM4juncap != 2)
        return;
    JC2initInst(&mod->BSIM4jc2td, 0, &here->BSIM4jc2i[0], Aseff, Pseff, LG,
                here->BSIM4jc2ifactor, here->BSIM4jc2cfactor);
    JC2initInst(&mod->BSIM4jc2td, 1, &here->BSIM4jc2i[1], Adeff, Pdeff, LG,
                here->BSIM4jc2ifactor, here->BSIM4jc2cfactor);
}
