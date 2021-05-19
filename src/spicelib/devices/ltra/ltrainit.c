#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "ltraitf.h"
#include "ltraext.h"
#include "ltrainit.h"


SPICEdev LTRAinfo = {
    .DEVpublic = {
        .name = "LTRA",
        .description = "Lossy transmission line",
        .terms = &LTRAnSize,
        .numNames = &LTRAnSize,
        .termNames = LTRAnames,
        .numInstanceParms = &LTRApTSize,
        .instanceParms = LTRApTable,
        .numModelParms = &LTRAmPTSize,
        .modelParms = LTRAmPTable,
        .flags = 0,

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

    .DEVparam = LTRAparam,
    .DEVmodParam = LTRAmParam,
    .DEVload = LTRAload,
    .DEVsetup = LTRAsetup,
    .DEVunsetup = LTRAunsetup,
    .DEVpzSetup = LTRAsetup,
    .DEVtemperature = LTRAtemp,
    .DEVtrunc = LTRAtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = LTRAacLoad,
    .DEVaccept = LTRAaccept,
    .DEVdestroy = NULL,
    .DEVmodDelete = LTRAmDelete,
    .DEVdelete = LTRAdevDelete,
    .DEVsetic = NULL,
    .DEVask = LTRAask,
    .DEVmodAsk = LTRAmAsk,
    .DEVpzLoad = NULL,
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
    .DEVinstSize = &LTRAiSize,
    .DEVmodSize = &LTRAmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_ltra_info(void)
{
    return &LTRAinfo;
}
