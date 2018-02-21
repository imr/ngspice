#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "asrcitf.h"
#include "asrcext.h"
#include "asrcinit.h"


SPICEdev ASRCinfo = {
    .DEVpublic = {
        .name = "ASRC",
        .description = "Arbitrary Source ",
        .terms = &ASRCnSize,
        .numNames = &ASRCnSize,
        .termNames = ASRCnames,
        .numInstanceParms = &ASRCpTSize,
        .instanceParms = ASRCpTable,
        .numModelParms = NULL,
        .modelParms = NULL,
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
    .DEVparam = ASRCparam,
    .DEVmodParam = NULL,
    .DEVload = ASRCload,
    .DEVsetup = ASRCsetup,
    .DEVunsetup = ASRCunsetup,
    .DEVpzSetup = ASRCsetup,
    .DEVtemperature = ASRCtemp,
    .DEVtrunc = NULL,
    .DEVfindBranch = ASRCfindBr,
    .DEVacLoad = ASRCacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = ASRCdestroy,
    .DEVmodDelete = ASRCmDelete,
    .DEVdelete = ASRCdelete,
    .DEVsetic = NULL,
    .DEVask = ASRCask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = ASRCpzLoad,
    .DEVconvTest = ASRCconvTest,
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
    .DEVinstSize = &ASRCiSize,
    .DEVmodSize = &ASRCmSize,
};


SPICEdev *
get_asrc_info(void)
{
    return &ASRCinfo;
}
