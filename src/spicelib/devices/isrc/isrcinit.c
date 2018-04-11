#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "isrcitf.h"
#include "isrcext.h"
#include "isrcinit.h"


SPICEdev ISRCinfo = {
    .DEVpublic = {
        .name = "Isource",
        .description = "Independent current source",
        .terms = &ISRCnSize,
        .numNames = &ISRCnSize,
        .termNames = ISRCnames,
        .numInstanceParms = &ISRCpTSize,
        .instanceParms = ISRCpTable,
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

    .DEVparam = ISRCparam,
    .DEVmodParam = NULL,
    .DEVload = ISRCload,
    .DEVsetup = NULL,
    .DEVunsetup = NULL,
    .DEVpzSetup = NULL,
    .DEVtemperature = ISRCtemp,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = ISRCacLoad,
    .DEVaccept = ISRCaccept,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = ISRCdelete,
    .DEVsetic = NULL,
    .DEVask = ISRCask,
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
    .DEVinstSize = &ISRCiSize,
    .DEVmodSize = &ISRCmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_isrc_info(void)
{
    return &ISRCinfo;
}
