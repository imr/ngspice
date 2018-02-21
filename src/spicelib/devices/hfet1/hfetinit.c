#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hfetitf.h"
#include "hfetext.h"
#include "hfetinit.h"


SPICEdev HFETAinfo = {
    .DEVpublic = {
        .name = "HFET1",
        .description = "HFET1 Model",
        .terms = &HFETAnSize,
        .numNames = &HFETAnSize,
        .termNames = HFETAnames,
        .numInstanceParms = &HFETApTSize,
        .instanceParms = HFETApTable,
        .numModelParms = &HFETAmPTSize,
        .modelParms = HFETAmPTable,
#ifdef XSPICE
        .cm_func = NULL,
        .num_conn = 0,
        .conn = NULL,
        .num_param = 0,
        .param = NULL,
        .num_inst_var = 0,
        .inst_var = NULL,
#endif
        .flags = DEV_DEFAULT,
    },
    .DEVparam = HFETAparam,
    .DEVmodParam = HFETAmParam,
    .DEVload = HFETAload,
    .DEVsetup = HFETAsetup,
    .DEVunsetup = HFETAunsetup,
    .DEVpzSetup = HFETAsetup,
    .DEVtemperature = HFETAtemp,
    .DEVtrunc = HFETAtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HFETAacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = HFETAdestroy,
    .DEVmodDelete = HFETAmDelete,
    .DEVdelete = HFETAdelete,
    .DEVsetic = HFETAgetic,
    .DEVask = HFETAask,
    .DEVmodAsk = HFETAmAsk,
    .DEVpzLoad = HFETApzLoad,
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
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &HFETAiSize,
    .DEVmodSize = &HFETAmSize,
};


SPICEdev *
get_hfeta_info(void)
{
    return &HFETAinfo;
}
