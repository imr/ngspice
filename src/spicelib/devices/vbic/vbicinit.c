/*
 * vbicinit.c
 */


#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vbicitf.h"
#include "vbicext.h"
#include "vbicinit.h"


SPICEdev VBICinfo = {
    .DEVpublic = {
        .name = "VBIC",
        .description = "Vertical Bipolar Inter-Company Model",
        .terms = &VBICnSize,
        .numNames = &VBICnSize,
        .termNames = VBICnames,
        .numInstanceParms = &VBICpTSize,
        .instanceParms = VBICpTable,
        .numModelParms = &VBICmPTSize,
        .modelParms = VBICmPTable,
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

    .DEVparam = VBICparam,
    .DEVmodParam = VBICmParam,
    .DEVload = VBICload,
    .DEVsetup = VBICsetup,
    .DEVunsetup = VBICunsetup,
    .DEVpzSetup = VBICsetup,
    .DEVtemperature = VBICtemp,
    .DEVtrunc = VBICtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = VBICacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = VBICgetic,
    .DEVask = VBICask,
    .DEVmodAsk = VBICmAsk,
    .DEVpzLoad = VBICpzLoad,
    .DEVconvTest = VBICconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = VBICnoise,
    .DEVsoaCheck = VBICsoaCheck,
    .DEVinstSize = &VBICiSize,
    .DEVmodSize = &VBICmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_vbic_info(void)
{
    return &VBICinfo;
}
