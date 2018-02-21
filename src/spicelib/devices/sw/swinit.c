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
    .DEVdestroy = SWdestroy,
    .DEVmodDelete = SWmDelete,
    .DEVdelete = SWdelete,
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
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &SWiSize,
    .DEVmodSize = &SWmSize,
};


SPICEdev *
get_sw_info(void)
{
    return &SWinfo;
}
