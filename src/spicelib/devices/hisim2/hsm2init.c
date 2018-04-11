#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsm2def.h"
#include "hsm2itf.h"
#include "hsm2init.h"


SPICEdev HSM2info = {
    .DEVpublic = {
        .name = "HiSIM2",
        .description = "Hiroshima University STARC IGFET Model 2.8.0",
        .terms = &HSM2nSize,
        .numNames = &HSM2nSize,
        .termNames = HSM2names,
        .numInstanceParms = &HSM2pTSize,
        .instanceParms = HSM2pTable,
        .numModelParms = &HSM2mPTSize,
        .modelParms = HSM2mPTable,
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

    .DEVparam = HSM2param,
    .DEVmodParam = HSM2mParam,
    .DEVload = HSM2load,
    .DEVsetup = HSM2setup,
    .DEVunsetup = HSM2unsetup,
    .DEVpzSetup = HSM2setup,
    .DEVtemperature = HSM2temp,
    .DEVtrunc = HSM2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HSM2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = HSM2mDelete,
    .DEVdelete = NULL,
    .DEVsetic = HSM2getic,
    .DEVask = HSM2ask,
    .DEVmodAsk = HSM2mAsk,
    .DEVpzLoad = HSM2pzLoad,
    .DEVconvTest = HSM2convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = HSM2noise,
    .DEVsoaCheck = HSM2soaCheck,
    .DEVinstSize = &HSM2iSize,
    .DEVmodSize = &HSM2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_hsm2_info(void)
{
    return &HSM2info;
}
