#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vsrcitf.h"
#include "vsrcdefs.h"
#include "vsrcinit.h"


SPICEdev VSRCinfo = {
    .DEVpublic = {
        .name = "Vsource",
        .description = "Independent voltage source",
        .terms = &VSRCnSize,
        .numNames = &VSRCnSize,
        .termNames = VSRCnames,
        .numInstanceParms = &VSRCpTSize,
        .instanceParms = VSRCpTable,
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

    .DEVparam = VSRCparam,
    .DEVmodParam = NULL,
    .DEVload = VSRCload,
    .DEVsetup = VSRCsetup,
    .DEVunsetup = VSRCunsetup,
    .DEVpzSetup = VSRCpzSetup,
    .DEVtemperature = VSRCtemp,
    .DEVtrunc = NULL,
    .DEVfindBranch = VSRCfindBr,
    .DEVacLoad = VSRCacLoad,
    .DEVaccept = VSRCaccept,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = VSRCdelete,
    .DEVsetic = NULL,
    .DEVask = VSRCask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = VSRCpzLoad,
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
    .DEVinstSize = &VSRCiSize,
    .DEVmodSize = &VSRCmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_vsrc_info(void)
{
    return &VSRCinfo;
}
