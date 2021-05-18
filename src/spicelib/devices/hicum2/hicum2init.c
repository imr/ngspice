/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/


#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hicum2itf.h"
#include "hicum2ext.h"
#include "hicum2init.h"
#include "hicumL2.hpp"
#include "hicumL2temp.hpp"


SPICEdev HICUMinfo = {
    .DEVpublic = {
        .name = "hicum2",
        .description = "High Current Model for BJT",
        .terms = &HICUMnSize,
        .numNames = &HICUMnSize,
        .termNames = HICUMnames,
        .numInstanceParms = &HICUMpTSize,
        .instanceParms = HICUMpTable,
        .numModelParms = &HICUMmPTSize,
        .modelParms = HICUMmPTable,
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

    .DEVparam = HICUMparam,
    .DEVmodParam = HICUMmParam,
    .DEVload = HICUMload,
    .DEVsetup = HICUMsetup,
    .DEVunsetup = HICUMunsetup,
    .DEVpzSetup = HICUMsetup,
    .DEVtemperature = HICUMtemp,
    .DEVtrunc = HICUMtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HICUMacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = HICUMmDelete,
    .DEVdelete = NULL,
    .DEVsetic = HICUMgetic,
    .DEVask = HICUMask,
    .DEVmodAsk = HICUMmAsk,
    .DEVpzLoad = HICUMpzLoad,
    .DEVconvTest = HICUMconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = HICUMnoise,
    .DEVsoaCheck = HICUMsoaCheck,
    .DEVinstSize = &HICUMiSize,
    .DEVmodSize = &HICUMmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_hicum_info(void)
{
    return &HICUMinfo;
}
