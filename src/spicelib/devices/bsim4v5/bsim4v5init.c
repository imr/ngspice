#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v5itf.h"
#include "bsim4v5ext.h"
#include "bsim4v5init.h"


SPICEdev BSIM4v5info = {
    .DEVpublic = {
        .name = "BSIM4v5",
        .description = "Berkeley Short Channel IGFET Model-4",
        .terms = &BSIM4v5nSize,
        .numNames = &BSIM4v5nSize,
        .termNames = BSIM4v5names,
        .numInstanceParms = &BSIM4v5pTSize,
        .instanceParms = BSIM4v5pTable,
        .numModelParms = &BSIM4v5mPTSize,
        .modelParms = BSIM4v5mPTable,
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

    .DEVparam = BSIM4v5param,
    .DEVmodParam = BSIM4v5mParam,
    .DEVload = BSIM4v5load,
    .DEVsetup = BSIM4v5setup,
    .DEVunsetup = BSIM4v5unsetup,
    .DEVpzSetup = BSIM4v5setup,
    .DEVtemperature = BSIM4v5temp,
    .DEVtrunc = BSIM4v5trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM4v5acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM4v5mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM4v5getic,
    .DEVask = BSIM4v5ask,
    .DEVmodAsk = BSIM4v5mAsk,
    .DEVpzLoad = BSIM4v5pzLoad,
    .DEVconvTest = BSIM4v5convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM4v5noise,
    .DEVsoaCheck = BSIM4v5soaCheck,
    .DEVinstSize = &BSIM4v5iSize,
    .DEVmodSize = &BSIM4v5mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim4v5_info(void)
{
    return &BSIM4v5info;
}
