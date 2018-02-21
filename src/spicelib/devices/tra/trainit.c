#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "traitf.h"
#include "traext.h"
#include "trainit.h"


SPICEdev TRAinfo = {
    .DEVpublic = {
        .name = "Tranline",
        .description = "Lossless transmission line",
        .terms = &TRAnSize,
        .numNames = &TRAnSize,
        .termNames = TRAnames,
        .numInstanceParms = &TRApTSize,
        .instanceParms = TRApTable,
        .numModelParms = NULL,
        .modelParms = NULL,
#ifdef XSPICE
        .cm_func = NULL,
        .num_conn = 0,
        .conn = NULL,
        .num_param = 0,
        .param = NULL,
        .num_inst_var = 0,
        .inst_var = NULL,
#endif
        .flags = 0,
    },
    .DEVparam = TRAparam,
    .DEVmodParam = NULL,
    .DEVload = TRAload,
    .DEVsetup = TRAsetup,
    .DEVunsetup = TRAunsetup,
    .DEVpzSetup = TRAsetup,
    .DEVtemperature = TRAtemp,
    .DEVtrunc = TRAtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = TRAacLoad,
    .DEVaccept = TRAaccept,
    .DEVdestroy = TRAdestroy,
    .DEVmodDelete = TRAmDelete,
    .DEVdelete = TRAdelete,
    .DEVsetic = NULL,
    .DEVask = TRAask,
    .DEVmodAsk = NULL,
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
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &TRAiSize,
    .DEVmodSize = &TRAmSize,
};


SPICEdev *
get_tra_info(void)
{
    return &TRAinfo;
}
