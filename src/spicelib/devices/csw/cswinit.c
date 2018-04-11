/* Modified: 2000 AlansFixes */

#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "cswitf.h"
#include "cswext.h"
#include "cswinit.h"


SPICEdev CSWinfo = {
    .DEVpublic = {
        .name = "CSwitch",
        .description = "Current controlled ideal switch",
        .terms = &CSWnSize,
        .numNames = &CSWnSize,
        .termNames = CSWnames,
        .numInstanceParms = &CSWpTSize,
        .instanceParms = CSWpTable,
        .numModelParms = &CSWmPTSize,
        .modelParms = CSWmPTable,
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

    .DEVparam = CSWparam,
    .DEVmodParam = CSWmParam,
    .DEVload = CSWload,
    .DEVsetup = CSWsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = CSWsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = CSWtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = CSWacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = CSWask,
    .DEVmodAsk = CSWmAsk,
    .DEVpzLoad = CSWpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = CSWnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &CSWiSize,
    .DEVmodSize = &CSWmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_csw_info(void)
{
    return &CSWinfo;
}
