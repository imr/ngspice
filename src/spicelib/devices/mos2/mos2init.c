#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mos2itf.h"
#include "mos2ext.h"
#include "mos2init.h"


SPICEdev MOS2info = {
    .DEVpublic = {
        .name = "Mos2",
        .description = "Level 2 MOSfet model with Meyer capacitance model",
        .terms = &MOS2nSize,
        .numNames = &MOS2nSize,
        .termNames = MOS2names,
        .numInstanceParms = &MOS2pTSize,
        .instanceParms = MOS2pTable,
        .numModelParms = &MOS2mPTSize,
        .modelParms = MOS2mPTable,
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

    .DEVparam = MOS2param,
    .DEVmodParam = MOS2mParam,
    .DEVload = MOS2load,
    .DEVsetup = MOS2setup,
    .DEVunsetup = MOS2unsetup,
    .DEVpzSetup = MOS2setup,
    .DEVtemperature = MOS2temp,
    .DEVtrunc = MOS2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MOS2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = MOS2delete,
    .DEVsetic = MOS2getic,
    .DEVask = MOS2ask,
    .DEVmodAsk = MOS2mAsk,
    .DEVpzLoad = MOS2pzLoad,
    .DEVconvTest = MOS2convTest,
    .DEVsenSetup = MOS2sSetup,
    .DEVsenLoad = MOS2sLoad,
    .DEVsenUpdate = MOS2sUpdate,
    .DEVsenAcLoad = MOS2sAcLoad,
    .DEVsenPrint = MOS2sPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = MOS2disto,
    .DEVnoise = MOS2noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MOS2iSize,
    .DEVmodSize = &MOS2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mos2_info(void)
{
    return &MOS2info;
}
