/* Modified: Alansfixes */
#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "switf.h"
#include "swext.h"
#include "swinit.h"


SPICEdev SWinfo = {
    .DEVpublic = {
        .name = "Switch",
        .description = "Ideal voltage controlled switch",
        .terms = &SWnSize,
        .numNames = &SWnSize,
        .termNames = SWnames,
        .numInstanceParms = &SWpTSize,
        .instanceParms = SWpTable,
        .numModelParms = &SWmPTSize,
        .modelParms = SWmPTable,
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

    .DEVparam = SWparam,
    .DEVmodParam = SWmParam,
    .DEVload = SWload,
    .DEVsetup = SWsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = SWsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = SWtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = SWacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = SWask,
    .DEVmodAsk = SWmAsk,
    .DEVpzLoad = SWpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = SWnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &SWiSize,
    .DEVmodSize = &SWmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_sw_info(void)
{
    return &SWinfo;
}
