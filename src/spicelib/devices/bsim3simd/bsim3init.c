#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3itf.h"
#include "bsim3ext.h"
#include "bsim3init.h"


SPICEdev BSIM3SIMDinfo = {
    .DEVpublic = {
        .name = "BSIM3simd",
        .description = "Berkeley Short Channel IGFET Model Version-3",
        .terms = &BSIM3SIMDnSize,
        .numNames = &BSIM3SIMDnSize,
        .termNames = BSIM3SIMDnames,
        .numInstanceParms = &BSIM3SIMDpTSize,
        .instanceParms = BSIM3SIMDpTable,
        .numModelParms = &BSIM3SIMDmPTSize,
        .modelParms = BSIM3SIMDmPTable,
        .flags = DEV_DEFAULT,

#ifdef XSPICE
        .cm_func = NULL,
        .num_conn = 0,
        .conn = NULL,
        .num_param = 0,
        .param = NULL,
        .num_inst_var = 0,
        .inst_var = NULL,
#endif
    },

    .DEVparam = BSIM3SIMDparam,
    .DEVmodParam = BSIM3SIMDmParam,
    .DEVload = BSIM3SIMDloadSel,
    .DEVsetup = BSIM3SIMDsetup,
    .DEVunsetup = BSIM3SIMDunsetup,
    .DEVpzSetup = BSIM3SIMDsetup,
    .DEVtemperature = BSIM3SIMDtemp,
    .DEVtrunc = BSIM3SIMDtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3SIMDacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM3SIMDmDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3SIMDgetic,
    .DEVask = BSIM3SIMDask,
    .DEVmodAsk = BSIM3SIMDmAsk,
    .DEVpzLoad = BSIM3SIMDpzLoad,
    .DEVconvTest = BSIM3SIMDconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3SIMDnoise,
    .DEVsoaCheck = BSIM3SIMDsoaCheck,
    .DEVinstSize = &BSIM3SIMDiSize,
    .DEVmodSize = &BSIM3SIMDmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3simd_info(void)
{
    return &BSIM3SIMDinfo;
}


