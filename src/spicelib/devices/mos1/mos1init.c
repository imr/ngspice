#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mos1itf.h"
#include "mos1ext.h"
#include "mos1init.h"


SPICEdev MOS1info = {
    .DEVpublic = {
        .name = "Mos1",
        .description = "Level 1 MOSfet model with Meyer capacitance model",
        .terms = &MOS1nSize,
        .numNames = &MOS1nSize,
        .termNames = MOS1names,
        .numInstanceParms = &MOS1pTSize,
        .instanceParms = MOS1pTable,
        .numModelParms = &MOS1mPTSize,
        .modelParms = MOS1mPTable,
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

    .DEVparam = MOS1param,
    .DEVmodParam = MOS1mParam,
    .DEVload = MOS1load,
    .DEVsetup = MOS1setup,
    .DEVunsetup = MOS1unsetup,
    .DEVpzSetup = MOS1setup,
    .DEVtemperature = MOS1temp,
    .DEVtrunc = MOS1trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MOS1acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = MOS1delete,
    .DEVsetic = MOS1getic,
    .DEVask = MOS1ask,
    .DEVmodAsk = MOS1mAsk,
    .DEVpzLoad = MOS1pzLoad,
    .DEVconvTest = MOS1convTest,
    .DEVsenSetup = MOS1sSetup,
    .DEVsenLoad = MOS1sLoad,
    .DEVsenUpdate = MOS1sUpdate,
    .DEVsenAcLoad = MOS1sAcLoad,
    .DEVsenPrint = MOS1sPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = MOS1disto,
    .DEVnoise = MOS1noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MOS1iSize,
    .DEVmodSize = &MOS1mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mos1_info(void)
{
    return &MOS1info;
}
