#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "inditf.h"
#include "indext.h"
#include "indinit.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

SPICEdev INDinfo = {
    .DEVpublic = {
        .name = "Inductor",
        .description = "Fixed inductor",
        .terms = &INDnSize,
        .numNames = &INDnSize,
        .termNames = INDnames,
        .numInstanceParms = &INDpTSize,
        .instanceParms = INDpTable,
        .numModelParms = &INDmPTSize,
        .modelParms = INDmPTable,
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

    .DEVparam = INDparam,
    .DEVmodParam = INDmParam,
#ifdef USE_CUSPICE
    .DEVload = cuINDload,
#else
    .DEVload = INDload,
#endif
    .DEVsetup = INDsetup,
    .DEVunsetup = INDunsetup,
    .DEVpzSetup = INDsetup,
    .DEVtemperature = INDtemp,
#ifdef USE_CUSPICE
    .DEVtrunc = cuINDtrunc,
#else
    .DEVtrunc = INDtrunc,
#endif
    .DEVfindBranch = NULL,
    .DEVacLoad = INDacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = INDask,
    .DEVmodAsk = INDmAsk,
    .DEVpzLoad = INDpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = INDsSetup,
    .DEVsenLoad = INDsLoad,
    .DEVsenUpdate = INDsUpdate,
    .DEVsenAcLoad = INDsAcLoad,
    .DEVsenPrint = INDsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &INDiSize,
    .DEVmodSize = &INDmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
#ifdef KLU
    .DEVbindCSC = INDbindCSC,
    .DEVbindCSCComplex = INDbindCSCComplex,
    .DEVbindCSCComplexToReal = INDbindCSCComplexToReal,
#endif
#ifdef USE_CUSPICE
    .cuDEVdestroy = cuINDdestroy,
    .DEVtopology = INDtopology,
#endif
};


SPICEdev MUTinfo = {
    .DEVpublic = {
        .name = "mutual",
        .description = "Mutual inductors",
        .terms = NULL,
        .numNames = NULL,
        .termNames = NULL,
        .numInstanceParms = &MUTpTSize,
        .instanceParms = MUTpTable,
        .numModelParms = NULL,
        .modelParms = NULL,
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

    .DEVparam = MUTparam,
    .DEVmodParam = NULL,
#ifdef USE_CUSPICE
    .DEVload = cuMUTload,
#else
    .DEVload = NULL,
#endif
    .DEVsetup = MUTsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = MUTsetup,
    .DEVtemperature = MUTtemp,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = MUTacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = MUTask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = MUTpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = MUTsSetup,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = MUTsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MUTiSize,
    .DEVmodSize = &MUTmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
#ifdef KLU
    .DEVbindCSC = MUTbindCSC,
    .DEVbindCSCComplex = MUTbindCSCComplex,
    .DEVbindCSCComplexToReal = MUTbindCSCComplexToReal,
#endif
#ifdef USE_CUSPICE
    .cuDEVdestroy = cuMUTdestroy,
    .DEVtopology = MUTtopology,
#endif
};


SPICEdev *
get_ind_info(void)
{
    return &INDinfo;
}


SPICEdev *
get_mut_info(void)
{
    return &MUTinfo;
}
