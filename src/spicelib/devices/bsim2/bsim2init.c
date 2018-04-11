#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim2itf.h"
#include "bsim2ext.h"
#include "bsim2init.h"


SPICEdev B2info = {
    .DEVpublic = {
        .name = "BSIM2",
        .description = "Berkeley Short Channel IGFET Model",
        .terms = &B2nSize,
        .numNames = &B2nSize,
        .termNames = B2names,
        .numInstanceParms = &B2pTSize,
        .instanceParms = B2pTable,
        .numModelParms = &B2mPTSize,
        .modelParms = B2mPTable,
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

    .DEVparam = B2param,
    .DEVmodParam = B2mParam,
    .DEVload = B2load,
    .DEVsetup = B2setup,
    .DEVunsetup = B2unsetup,
    .DEVpzSetup = B2setup,
    .DEVtemperature = B2temp,
    .DEVtrunc = B2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = B2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = B2getic,
    .DEVask = B2ask,
    .DEVmodAsk = B2mAsk,
    .DEVpzLoad = B2pzLoad,
    .DEVconvTest = B2convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = B2noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &B2iSize,
    .DEVmodSize = &B2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim2_info(void)
{
    return &B2info;
}
