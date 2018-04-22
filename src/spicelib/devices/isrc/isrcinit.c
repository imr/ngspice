#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "isrcitf.h"
#include "isrcext.h"
#include "isrcinit.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

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
#ifdef USE_CUSPICE
    .DEVload = cuISRCload,
#else
    .DEVload = ISRCload,
#endif
#ifdef USE_CUSPICE
    .DEVsetup = ISRCsetup,
#else
    .DEVsetup = NULL,
#endif
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
#ifdef KLU
    .DEVbindCSC = NULL,
    .DEVbindCSCComplex = NULL,
    .DEVbindCSCComplexToReal = NULL,
#endif
#ifdef USE_CUSPICE
    .cuDEVdestroy = cuISRCdestroy,
    .DEVtopology = ISRCtopology,
#endif
};


SPICEdev *
get_isrc_info(void)
{
    return &ISRCinfo;
}
