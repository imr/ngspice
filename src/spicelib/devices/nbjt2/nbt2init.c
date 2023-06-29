#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "nbjt2itf.h"
#include "nbjt2ext.h"
#include "nbt2init.h"


SPICEdev NBJT2info = {
    .DEVpublic = {
        .name = "NBJT2",
        .description = "2D Numerical Bipolar Junction Transistor model",
        .terms = &NBJT2nSize,
        .numNames = &NBJT2nSize,
        .termNames = NBJT2names,
        .numInstanceParms = &NBJT2pTSize,
        .instanceParms = NBJT2pTable,
        .numModelParms = &NBJT2mPTSize,
        .modelParms = NBJT2mPTable,
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

    .DEVparam = NBJT2param,
    .DEVmodParam = NBJT2mParam,
    .DEVload = NBJT2load,
    .DEVsetup = NBJT2setup,
    .DEVunsetup = NULL,
    .DEVpzSetup = NBJT2setup,
    .DEVtemperature = NBJT2temp,
    .DEVtrunc = NBJT2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NBJT2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NBJT2modDelete,
    .DEVdelete = NBJT2delete,
    .DEVsetic = NULL,
    .DEVask = NBJT2ask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NBJT2pzLoad,
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
    .DEVinstSize = &NBJT2iSize,
    .DEVmodSize = &NBJT2mSize,

#ifdef CIDER
    .DEVdump = NBJT2dump,
    .DEVacct = NBJT2acct,
#endif
};


SPICEdev *
get_nbjt2_info(void)
{
    return &NBJT2info;
}
