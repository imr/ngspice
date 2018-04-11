#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mos6itf.h"
#include "mos6ext.h"
#include "mos6init.h"


SPICEdev MOS6info = {
    .DEVpublic = {
        .name = "Mos6",
        .description = "Level 6 MOSfet model with Meyer capacitance model",
        .terms = &MOS6nSize,
        .numNames = &MOS6nSize,
        .termNames = MOS6names,
        .numInstanceParms = &MOS6pTSize,
        .instanceParms = MOS6pTable,
        .numModelParms = &MOS6mPTSize,
        .modelParms = MOS6mPTable,
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

    .DEVparam = MOS6param,
    .DEVmodParam = MOS6mParam,
    .DEVload = MOS6load,
    .DEVsetup = MOS6setup,
    .DEVunsetup = MOS6unsetup,
    .DEVpzSetup = NULL,
    .DEVtemperature = MOS6temp,
    .DEVtrunc = MOS6trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NULL,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = MOS6delete,
    .DEVsetic = MOS6getic,
    .DEVask = MOS6ask,
    .DEVmodAsk = MOS6mAsk,
    .DEVpzLoad = NULL,
    .DEVconvTest = MOS6convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MOS6iSize,
    .DEVmodSize = &MOS6mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mos6_info(void)
{
    return &MOS6info;
}
