#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "b4soiitf.h"
#include "b4soiinit.h"

SPICEdev B4SOIinfo = {
    .DEVpublic = {
        .name = "B4SOI",
        .description = "Berkeley SOI MOSFET model version 4.4.0",
        .terms = &B4SOInSize,
        .numNames = &B4SOInSize,
        .termNames = B4SOInames,
        .numInstanceParms = &B4SOIpTSize,
        .instanceParms = B4SOIpTable,
        .numModelParms = &B4SOImPTSize,
        .modelParms = B4SOImPTable,
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

    .DEVparam = B4SOIparam,
    .DEVmodParam = B4SOImParam,
    .DEVload = B4SOIload,
    .DEVsetup = B4SOIsetup,
    .DEVunsetup = B4SOIunsetup,
    .DEVpzSetup = B4SOIsetup,
    .DEVtemperature = B4SOItemp,
    .DEVtrunc = B4SOItrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = B4SOIacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = B4SOImDelete,
    .DEVdelete = NULL,
    .DEVsetic = B4SOIgetic,
    .DEVask = B4SOIask,
    .DEVmodAsk = B4SOImAsk,
    .DEVpzLoad = B4SOIpzLoad,
    .DEVconvTest = B4SOIconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = B4SOInoise,
    .DEVsoaCheck = B4SOIsoaCheck,
    .DEVinstSize = &B4SOIiSize,
    .DEVmodSize = &B4SOImSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_b4soi_info (void)
{
  return &B4SOIinfo;
}


