#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v6itf.h"
#include "bsim4v6ext.h"
#include "bsim4v6init.h"


SPICEdev BSIM4v6info = {
    .DEVpublic = {
        .name = "BSIM4v6",
        .description = "Berkeley Short Channel IGFET Model-4",
        .terms = &BSIM4v6nSize,
        .numNames = &BSIM4v6nSize,
        .termNames = BSIM4v6names,
        .numInstanceParms = &BSIM4v6pTSize,
        .instanceParms = BSIM4v6pTable,
        .numModelParms = &BSIM4v6mPTSize,
        .modelParms = BSIM4v6mPTable,
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

    .DEVparam = BSIM4v6param,
    .DEVmodParam = BSIM4v6mParam,
    .DEVload = BSIM4v6load,
    .DEVsetup = BSIM4v6setup,
    .DEVunsetup = BSIM4v6unsetup,
    .DEVpzSetup = BSIM4v6setup,
    .DEVtemperature = BSIM4v6temp,
    .DEVtrunc = BSIM4v6trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM4v6acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM4v6mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM4v6getic,
    .DEVask = BSIM4v6ask,
    .DEVmodAsk = BSIM4v6mAsk,
    .DEVpzLoad = BSIM4v6pzLoad,
    .DEVconvTest = BSIM4v6convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM4v6noise,
    .DEVsoaCheck = BSIM4v6soaCheck,
    .DEVinstSize = &BSIM4v6iSize,
    .DEVmodSize = &BSIM4v6mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim4v6_info(void)
{
    return &BSIM4v6info;
}
