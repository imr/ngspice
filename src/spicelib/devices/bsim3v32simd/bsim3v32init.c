#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3v32itf.h"
#include "bsim3v32ext.h"
#include "bsim3v32init.h"


SPICEdev BSIM3v32simdinfo = {
    .DEVpublic = {
        .name = "BSIM3v32simd",
        .description = "Berkeley Short Channel IGFET Model Version-3",
        .terms = &BSIM3v32SIMDnSize,
        .numNames = &BSIM3v32SIMDnSize,
        .termNames = BSIM3v32SIMDnames,
        .numInstanceParms = &BSIM3v32SIMDpTSize,
        .instanceParms = BSIM3v32SIMDpTable,
        .numModelParms = &BSIM3v32SIMDmPTSize,
        .modelParms = BSIM3v32SIMDmPTable,
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

    .DEVparam = BSIM3v32SIMDparam,
    .DEVmodParam = BSIM3v32SIMDmParam,
#ifdef BSIM3v32SIMD
    .DEVload = BSIM3v32SIMDloadSel, /*F.B: point to load function wrapper */
#else
    .DEVload = BSIM3v32SIMDload,
#endif
    .DEVsetup = BSIM3v32SIMDsetup,
    .DEVunsetup = BSIM3v32SIMDunsetup,
    .DEVpzSetup = BSIM3v32SIMDsetup,
    .DEVtemperature = BSIM3v32SIMDtemp,
    .DEVtrunc = BSIM3v32SIMDtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3v32SIMDacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM3v32SIMDmDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3v32SIMDgetic,
    .DEVask = BSIM3v32SIMDask,
    .DEVmodAsk = BSIM3v32SIMDmAsk,
    .DEVpzLoad = BSIM3v32SIMDpzLoad,
    .DEVconvTest = BSIM3v32SIMDconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3v32SIMDnoise,
    .DEVsoaCheck = BSIM3v32SIMDsoaCheck,
    .DEVinstSize = &BSIM3v32SIMDiSize,
    .DEVmodSize = &BSIM3v32SIMDmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3v32simd_info(void)
{
    return &BSIM3v32simdinfo;
}
