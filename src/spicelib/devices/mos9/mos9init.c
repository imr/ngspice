#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mos9itf.h"
#include "mos9ext.h"
#include "mos9init.h"


SPICEdev MOS9info = {
    .DEVpublic = {
        .name = "Mos9",
        .description = "Modified Level 3 MOSfet model",
        .terms = &MOS9nSize,
        .numNames = &MOS9nSize,
        .termNames = MOS9names,
        .numInstanceParms = &MOS9pTSize,
        .instanceParms = MOS9pTable,
        .numModelParms = &MOS9mPTSize,
        .modelParms = MOS9mPTable,
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

    .DEVparam = MOS9param,
    .DEVmodParam = MOS9mParam,
    .DEVload = MOS9load,
    .DEVsetup = MOS9setup,
    .DEVunsetup = MOS9unsetup,
    .DEVpzSetup = MOS9setup,
    .DEVtemperature = MOS9temp,
    .DEVtrunc = MOS9trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MOS9acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = MOS9delete,
    .DEVsetic = MOS9getic,
    .DEVask = MOS9ask,
    .DEVmodAsk = MOS9mAsk,
    .DEVpzLoad = MOS9pzLoad,
    .DEVconvTest = MOS9convTest,
    .DEVsenSetup = MOS9sSetup,
    .DEVsenLoad = MOS9sLoad,
    .DEVsenUpdate = MOS9sUpdate,
    .DEVsenAcLoad = MOS9sAcLoad,
    .DEVsenPrint = MOS9sPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = MOS9disto,
    .DEVnoise = MOS9noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MOS9iSize,
    .DEVmodSize = &MOS9mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mos9_info(void)
{
    return &MOS9info;
}
