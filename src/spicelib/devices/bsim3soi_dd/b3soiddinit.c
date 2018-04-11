#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "b3soidditf.h"
#include "b3soiddinit.h"

SPICEdev B3SOIDDinfo = {
    .DEVpublic = {
        .name = "B3SOIDD",
        .description = "Berkeley SOI MOSFET (DD) model version 2.1",
        .terms = &B3SOIDDnSize,
        .numNames = &B3SOIDDnSize,
        .termNames = B3SOIDDnames,
        .numInstanceParms = &B3SOIDDpTSize,
        .instanceParms = B3SOIDDpTable,
        .numModelParms = &B3SOIDDmPTSize,
        .modelParms = B3SOIDDmPTable,
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

    .DEVparam = B3SOIDDparam,
    .DEVmodParam = B3SOIDDmParam,
    .DEVload = B3SOIDDload,
    .DEVsetup = B3SOIDDsetup,
    .DEVunsetup = B3SOIDDunsetup,
    .DEVpzSetup = B3SOIDDsetup,
    .DEVtemperature = B3SOIDDtemp,
    .DEVtrunc = B3SOIDDtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = B3SOIDDacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = B3SOIDDgetic,
    .DEVask = B3SOIDDask,
    .DEVmodAsk = B3SOIDDmAsk,
    .DEVpzLoad = B3SOIDDpzLoad,
    .DEVconvTest = B3SOIDDconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = B3SOIDDnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &B3SOIDDiSize,
    .DEVmodSize = &B3SOIDDmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_b3soidd_info (void)
{
  return &B3SOIDDinfo;
}


