#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vcvsitf.h"
#include "vcvsext.h"
#include "vcvsinit.h"


SPICEdev VCVSinfo = {
    .DEVpublic = {
        .name = "VCVS",
        .description = "Voltage controlled voltage source",
        .terms = &VCVSnSize,
        .numNames = &VCVSnSize,
        .termNames = VCVSnames,
        .numInstanceParms = &VCVSpTSize,
        .instanceParms = VCVSpTable,
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

    .DEVparam = VCVSparam,
    .DEVmodParam = NULL,
    .DEVload = VCVSload,
    .DEVsetup = VCVSsetup,
    .DEVunsetup = VCVSunsetup,
    .DEVpzSetup = VCVSsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = VCVSfindBr,
    .DEVacLoad = VCVSload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = VCVSask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = VCVSpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = VCVSsSetup,
    .DEVsenLoad = VCVSsLoad,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = VCVSsAcLoad,
    .DEVsenPrint = VCVSsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &VCVSiSize,
    .DEVmodSize = &VCVSmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_vcvs_info(void)
{
    return &VCVSinfo;
}
