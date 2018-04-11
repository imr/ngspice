#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4itf.h"
#include "bsim4ext.h"
#include "bsim4init.h"


SPICEdev BSIM4info = {
    .DEVpublic = {
        .name = "BSIM4",
        .description = "Berkeley Short Channel IGFET Model-4",
        .terms = &BSIM4nSize,
        .numNames = &BSIM4nSize,
        .termNames = BSIM4names,
        .numInstanceParms = &BSIM4pTSize,
        .instanceParms = BSIM4pTable,
        .numModelParms = &BSIM4mPTSize,
        .modelParms = BSIM4mPTable,
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

    .DEVparam = BSIM4param,
    .DEVmodParam = BSIM4mParam,
    .DEVload = BSIM4load,
    .DEVsetup = BSIM4setup,
    .DEVunsetup = BSIM4unsetup,
    .DEVpzSetup = BSIM4setup,
    .DEVtemperature = BSIM4temp,
    .DEVtrunc = BSIM4trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM4acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM4mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM4getic,
    .DEVask = BSIM4ask,
    .DEVmodAsk = BSIM4mAsk,
    .DEVpzLoad = BSIM4pzLoad,
    .DEVconvTest = BSIM4convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM4noise,
    .DEVsoaCheck = BSIM4soaCheck,
    .DEVinstSize = &BSIM4iSize,
    .DEVmodSize = &BSIM4mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim4_info(void)
{
    return &BSIM4info;
}
