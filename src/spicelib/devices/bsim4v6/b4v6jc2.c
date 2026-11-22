/**********
 * BSIM4.6: JUNCAP2 junction model (juncap = 2)
 *
 * Glue to the shared JUNCAP2 code in devices/juncap2.c, as in bsim4/b4jcap.c.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/const.h"

/* default JUNCAP2 reference temperature [K] (TRJ not given), as in other
   simulators: TNOM if given in the model card, else 27 C; the option TNOM
   has no effect */
static double
BSIM4v6jc2trdef(BSIM4v6model *mod)
{
    return mod->BSIM4v6tnomGiven ? mod->BSIM4v6tnom : CONSTCtoK + 27.0;
}

int
BSIM4v6jc2mParam(int param, IFvalue *value, BSIM4v6model *mod)
{
    if (param == BSIM4v6_MOD_JUNCAP) {
        mod->BSIM4v6juncap = value->iValue;
        mod->BSIM4v6juncapGiven = TRUE;
        return OK;
    }
    if (param >= BSIM4v6_MOD_JC2BASE && param < BSIM4v6_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mParam(&mod->BSIM4v6jc2, param - BSIM4v6_MOD_JC2BASE, value);
    return E_BADPARM;
}

int
BSIM4v6jc2mAsk(BSIM4v6model *mod, int which, IFvalue *value)
{
    if (which == BSIM4v6_MOD_JUNCAP) {
        value->iValue = mod->BSIM4v6juncap;
        return OK;
    }
    if (which >= BSIM4v6_MOD_JC2BASE && which < BSIM4v6_MOD_JC2BASE + JC2_MOD_NIDS)
        return JC2mAsk(&mod->BSIM4v6jc2, BSIM4v6jc2trdef(mod), which - BSIM4v6_MOD_JC2BASE, value);
    return E_BADPARM;
}

/* juncap selector and JUNCAP2 defaults, called from BSIM4v6setup() */
void
BSIM4v6jc2Setup(BSIM4v6model *mod)
{
    if (!mod->BSIM4v6juncapGiven) {
        mod->BSIM4v6juncap = 0;
    } else if (mod->BSIM4v6juncap == 1) {
        printf("Warning: juncap = 1 (JUNCAP1) is not implemented, juncap has been set to 0.\n");
        mod->BSIM4v6juncap = 0;
    } else if ((mod->BSIM4v6juncap != 0) && (mod->BSIM4v6juncap != 2)) {
        printf("Warning: juncap = %d not matched, juncap has been set to 0.\n", mod->BSIM4v6juncap);
        mod->BSIM4v6juncap = 0;
    }
    JC2setup(&mod->BSIM4v6jc2, mod->BSIM4v6modName, mod->BSIM4v6juncap == 2);
}

/* called from BSIM4v6temp() */
void
BSIM4v6jc2TempModel(BSIM4v6model *mod, double Temp)
{
    if (mod->BSIM4v6juncap == 2)
        JC2temp(&mod->BSIM4v6jc2, &mod->BSIM4v6jc2td, Temp, BSIM4v6jc2trdef(mod));
}

/* called from BSIM4v6temp() per instance, after the effective S/D geometry is
   known; Pseff/Pdeff without gate edge, LG = Weff_CJ * NF */
void
BSIM4v6jc2TempInst(BSIM4v6model *mod, BSIM4v6instance *here,
              double Aseff, double Pseff, double Adeff, double Pdeff, double LG)
{
    if (mod->BSIM4v6juncap != 2)
        return;
    JC2initInst(&mod->BSIM4v6jc2td, 0, &here->BSIM4v6jc2i[0], Aseff, Pseff, LG,
                here->BSIM4v6jc2ifactor, here->BSIM4v6jc2cfactor);
    JC2initInst(&mod->BSIM4v6jc2td, 1, &here->BSIM4v6jc2i[1], Adeff, Pdeff, LG,
                here->BSIM4v6jc2ifactor, here->BSIM4v6jc2cfactor);
}
