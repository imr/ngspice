#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "balunitf.h"
#include "balunext.h"
#include "baluninit.h"


SPICEdev BALUNinfo = {
    .DEVpublic = {
        .name = "BALUN",
        .description = "Ideal balun",
        .terms = &BALUNnSize,
        .numNames = &BALUNnSize,
        .termNames = BALUNnames,
        .numInstanceParms = &BALUNpTSize,
        .instanceParms = BALUNpTable,
        .numModelParms = NULL,
        .modelParms = NULL,
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

    .DEVparam = BALUNparam,
    .DEVmodParam = NULL,
    .DEVload = BALUNload,
    .DEVsetup = BALUNsetup,
    .DEVunsetup = BALUNunsetup,
    .DEVpzSetup = BALUNsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = BALUNload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = BALUNask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = BALUNpzLoad,
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
    .DEVinstSize = &BALUNiSize,
    .DEVmodSize = &BALUNmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_balun_info(void)
{
    return &BALUNinfo;
}
