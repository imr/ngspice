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
#ifdef XSPICE
        .cm_func = NULL,
        .num_conn = 0,
        .conn = NULL,
        .num_param = 0,
        .param = NULL,
        .num_inst_var = 0,
        .inst_var = NULL,
#endif
        .flags = DEV_DEFAULT,
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
    .DEVdestroy = NUMD2destroy,
    .DEVmodDelete = NUMD2mDelete,
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
#ifdef CIDER
    .DEVdump = NUMD2dump,
    .DEVacct = NUMD2acct,
#endif
    .DEVinstSize = &NUMD2iSize,
    .DEVmodSize = &NUMD2mSize,
};


SPICEdev *
get_numd2_info(void)
{
    return &NUMD2info;
}
