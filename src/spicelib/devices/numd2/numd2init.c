#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "numd2itf.h"
#include "numd2ext.h"
#include "numd2init.h"


SPICEdev NUMD2info = {
    .DEVpublic = {
        .name = "NUMD2",
        .description = "2D Numerical Junction Diode model",
        .terms = &NUMD2nSize,
        .numNames = &NUMD2nSize,
        .termNames = NUMD2names,
        .numInstanceParms = &NUMD2pTSize,
        .instanceParms = NUMD2pTable,
        .numModelParms = &NUMD2mPTSize,
        .modelParms = NUMD2mPTable,
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

    .DEVparam = NUMD2param,
    .DEVmodParam = NUMD2mParam,
    .DEVload = NUMD2load,
    .DEVsetup = NUMD2setup,
    .DEVunsetup = NULL,
    .DEVpzSetup = NUMD2setup,
    .DEVtemperature = NUMD2temp,
    .DEVtrunc = NUMD2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NUMD2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NUMD2modDelete,
    .DEVdelete = NUMD2delete,
    .DEVsetic = NULL,
    .DEVask = NUMD2ask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NUMD2pzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &NUMD2iSize,
    .DEVmodSize = &NUMD2mSize,

#ifdef CIDER
    .DEVdump = NUMD2dump,
    .DEVacct = NUMD2acct,
#endif
};


SPICEdev *
get_numd2_info(void)
{
    return &NUMD2info;
}
