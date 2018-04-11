#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vccsitf.h"
#include "vccsext.h"
#include "vccsinit.h"


SPICEdev VCCSinfo = {
    .DEVpublic = {
        .name = "VCCS",
        .description = "Voltage controlled current source",
        .terms = &VCCSnSize,
        .numNames = &VCCSnSize,
        .termNames = VCCSnames,
        .numInstanceParms = &VCCSpTSize,
        .instanceParms = VCCSpTable,
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

    .DEVparam = VCCSparam,
    .DEVmodParam = NULL,
    .DEVload = VCCSload,
    .DEVsetup = VCCSsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = VCCSsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = VCCSload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = VCCSask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = VCCSpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = VCCSsSetup,
    .DEVsenLoad = VCCSsLoad,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = VCCSsAcLoad,
    .DEVsenPrint = VCCSsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &VCCSiSize,
    .DEVmodSize = &VCCSmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_vccs_info(void)
{
    return &VCCSinfo;
}
