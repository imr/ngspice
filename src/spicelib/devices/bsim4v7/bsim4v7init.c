#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v7itf.h"
#include "bsim4v7ext.h"
#include "bsim4v7init.h"


SPICEdev BSIM4v7info = {
    .DEVpublic = {
        .name = "BSIM4v7",
        .description = "Berkeley Short Channel IGFET Model-4",
        .terms = &BSIM4v7nSize,
        .numNames = &BSIM4v7nSize,
        .termNames = BSIM4v7names,
        .numInstanceParms = &BSIM4v7pTSize,
        .instanceParms = BSIM4v7pTable,
        .numModelParms = &BSIM4v7mPTSize,
        .modelParms = BSIM4v7mPTable,
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

    .DEVparam = BSIM4v7param,
    .DEVmodParam = BSIM4v7mParam,
    .DEVload = BSIM4v7load,
    .DEVsetup = BSIM4v7setup,
    .DEVunsetup = BSIM4v7unsetup,
    .DEVpzSetup = BSIM4v7setup,
    .DEVtemperature = BSIM4v7temp,
    .DEVtrunc = BSIM4v7trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM4v7acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM4v7mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM4v7getic,
    .DEVask = BSIM4v7ask,
    .DEVmodAsk = BSIM4v7mAsk,
    .DEVpzLoad = BSIM4v7pzLoad,
    .DEVconvTest = BSIM4v7convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM4v7noise,
    .DEVsoaCheck = BSIM4v7soaCheck,
    .DEVinstSize = &BSIM4v7iSize,
    .DEVmodSize = &BSIM4v7mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim4v7_info(void)
{
    return &BSIM4v7info;
}
