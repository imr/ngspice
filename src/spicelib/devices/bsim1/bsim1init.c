#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim1itf.h"
#include "bsim1ext.h"
#include "bsim1init.h"


SPICEdev B1info = {
    .DEVpublic = {
        .name = "BSIM1",
        .description = "Berkeley Short Channel IGFET Model",
        .terms = &B1nSize,
        .numNames = &B1nSize,
        .termNames = B1names,
        .numInstanceParms = &B1pTSize,
        .instanceParms = B1pTable,
        .numModelParms = &B1mPTSize,
        .modelParms = B1mPTable,
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

    .DEVparam = B1param,
    .DEVmodParam = B1mParam,
    .DEVload = B1load,
    .DEVsetup = B1setup,
    .DEVunsetup = B1unsetup,
    .DEVpzSetup = B1setup,
    .DEVtemperature = B1temp,
    .DEVtrunc = B1trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = B1acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = B1getic,
    .DEVask = B1ask,
    .DEVmodAsk = B1mAsk,
    .DEVpzLoad = B1pzLoad,
    .DEVconvTest = B1convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = B1disto,
    .DEVnoise = B1noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &B1iSize,
    .DEVmodSize = &B1mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim1_info(void)
{
    return &B1info;
}
