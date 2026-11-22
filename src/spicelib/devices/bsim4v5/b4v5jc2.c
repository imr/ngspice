/**********
 * BSIM4.5: JUNCAP2 junction model (juncap = 2)
 *
 * Glue to the shared JUNCAP2 code in devices/juncap2.c, as in bsim4/b4jcap.c.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/const.h"

/* default JUNCAP2 reference temperature [K] (TRJ not given), as in other
   simulators: TNOM if given in the model card, else 27 C; the option TNOM
   has no effect */
static double
BSIM4v5jc2trdef(BSIM4v5model *mod)
{
    return mod->BSIM4v5tnomGiven ? mod->BSIM4v5tnom : CONSTCtoK + 27.0;
}

int
BSIM4v5jc2mParam(int param, IFvalue *value, BSIM4v5model *mod)
{
    if (param == BSIM4v5_MOD_JUNCAP) {
        mod->BSIM4v5juncap = value->iValue;
        mod->BSIM4v5juncapGiven = TRUE;
        return OK;
    }
    if (param >= BSIM4v5_MOD_JC2BASE && param < BSIM4v5_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mParam(&mod->BSIM4v5jc2, param - BSIM4v5_MOD_JC2BASE, value);
    return E_BADPARM;
}

int
BSIM4v5jc2mAsk(BSIM4v5model *mod, int which, IFvalue *value)
{
    if (which == BSIM4v5_MOD_JUNCAP) {
        value->iValue = mod->BSIM4v5juncap;
        return OK;
    }
    if (which >= BSIM4v5_MOD_JC2BASE && which < BSIM4v5_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mAsk(&mod->BSIM4v5jc2, BSIM4v5jc2trdef(mod), which - BSIM4v5_MOD_JC2BASE, value);
    return E_BADPARM;
}

/* juncap selector and JUNCAP2 defaults, called from BSIM4v5setup() */
void
BSIM4v5jc2Setup(BSIM4v5model *mod)
{
    if (!mod->BSIM4v5juncapGiven) {
        mod->BSIM4v5juncap = 0;
    } else if (mod->BSIM4v5juncap == 1) {
        printf("Warning: juncap = 1 (JUNCAP1) is not implemented, juncap has been set to 0.\n");
        mod->BSIM4v5juncap = 0;
    } else if ((mod->BSIM4v5juncap != 0) && (mod->BSIM4v5juncap != 2)) {
        printf("Warning: juncap = %d not matched, juncap has been set to 0.\n", mod->BSIM4v5juncap);
        mod->BSIM4v5juncap = 0;
    }
    JC2setup(&mod->BSIM4v5jc2, mod->BSIM4v5modName, mod->BSIM4v5juncap == 2);
}

/* called from BSIM4v5temp() */
void
BSIM4v5jc2TempModel(BSIM4v5model *mod, double Temp)
{
    if (mod->BSIM4v5juncap == 2)
        JC2temp(&mod->BSIM4v5jc2, &mod->BSIM4v5jc2td, Temp, BSIM4v5jc2trdef(mod));
}

/* called from BSIM4v5temp() per instance, after the effective S/D geometry is
   known; Pseff/Pdeff without gate edge, LG = Weff_CJ * NF */
void
BSIM4v5jc2TempInst(BSIM4v5model *mod, BSIM4v5instance *here,
              double Aseff, double Pseff, double Adeff, double Pdeff, double LG)
{
    if (mod->BSIM4v5juncap != 2)
        return;
    JC2initInst(&mod->BSIM4v5jc2td, 0, &here->BSIM4v5jc2i[0], Aseff, Pseff, LG,
                here->BSIM4v5jc2ifactor, here->BSIM4v5jc2cfactor);
    JC2initInst(&mod->BSIM4v5jc2td, 1, &here->BSIM4v5jc2i[1], Adeff, Pdeff, LG,
                here->BSIM4v5jc2ifactor, here->BSIM4v5jc2cfactor);
}
