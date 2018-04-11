#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "cccsitf.h"
#include "cccsext.h"
#include "cccsinit.h"


SPICEdev CCCSinfo = {
    .DEVpublic = {
        .name = "CCCS",
        .description = "Current controlled current source",
        .terms = &CCCSnSize,
        .numNames = &CCCSnSize,
        .termNames = CCCSnames,
        .numInstanceParms = &CCCSpTSize,
        .instanceParms = CCCSpTable,
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

    .DEVparam = CCCSparam,
    .DEVmodParam = NULL,
    .DEVload = CCCSload,
    .DEVsetup = CCCSsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = CCCSsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = CCCSload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = CCCSask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = CCCSpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = CCCSsSetup,
    .DEVsenLoad = CCCSsLoad,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = CCCSsAcLoad,
    .DEVsenPrint = CCCSsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &CCCSiSize,
    .DEVmodSize = &CCCSmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_cccs_info(void)
{
    return &CCCSinfo;
}
