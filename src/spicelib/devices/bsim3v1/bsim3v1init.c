#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3v1itf.h"
#include "bsim3v1ext.h"
#include "bsim3v1init.h"

SPICEdev BSIM3v1info = {
    .DEVpublic = {
        .name = "BSIM3v1",
        .description = "Berkeley Short Channel IGFET Model Version-3 (3.1)",
        .terms = &BSIM3v1nSize,
        .numNames = &BSIM3v1nSize,
        .termNames = BSIM3v1names,
        .numInstanceParms = &BSIM3v1pTSize,
        .instanceParms = BSIM3v1pTable,
        .numModelParms = &BSIM3v1mPTSize,
        .modelParms = BSIM3v1mPTable,
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

    .DEVparam = BSIM3v1param,
    .DEVmodParam = BSIM3v1mParam,
    .DEVload = BSIM3v1load,
    .DEVsetup = BSIM3v1setup,
    .DEVunsetup = BSIM3v1unsetup,
    .DEVpzSetup = BSIM3v1setup,
    .DEVtemperature = BSIM3v1temp,
    .DEVtrunc = BSIM3v1trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3v1acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3v1getic,
    .DEVask = BSIM3v1ask,
    .DEVmodAsk = BSIM3v1mAsk,
    .DEVpzLoad = BSIM3v1pzLoad,
    .DEVconvTest = BSIM3v1convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3v1noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &BSIM3v1iSize,
    .DEVmodSize = &BSIM3v1mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3v1_info(void)
{
     return &BSIM3v1info; 
}
