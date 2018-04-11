#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "ccvsitf.h"
#include "ccvsext.h"
#include "ccvsinit.h"


SPICEdev CCVSinfo = {
    .DEVpublic = {
        .name = "CCVS",
        .description = "Linear current controlled current source",
        .terms = &CCVSnSize,
        .numNames = &CCVSnSize,
        .termNames = CCVSnames,
        .numInstanceParms = &CCVSpTSize,
        .instanceParms = CCVSpTable,
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

    .DEVparam = CCVSparam,
    .DEVmodParam = NULL,
    .DEVload = CCVSload,
    .DEVsetup = CCVSsetup,
    .DEVunsetup = CCVSunsetup,
    .DEVpzSetup = CCVSsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = CCVSfindBr,
    .DEVacLoad = CCVSload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = CCVSask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = CCVSpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = CCVSsSetup,
    .DEVsenLoad = CCVSsLoad,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = CCVSsAcLoad,
    .DEVsenPrint = CCVSsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &CCVSiSize,
    .DEVmodSize = &CCVSmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_ccvs_info(void)
{
    return &CCVSinfo;
}
