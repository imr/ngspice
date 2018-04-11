#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3v0itf.h"
#include "bsim3v0ext.h"
#include "bsim3v0init.h"

SPICEdev B3v0info = {
    .DEVpublic = {
        .name = "BSIM3v0",
        .description = "Berkeley Short Channel IGFET Model Version-3 (3.0)",
        .terms = &BSIM3v0nSize,
        .numNames = &BSIM3v0nSize,
        .termNames = BSIM3v0names,
        .numInstanceParms = &BSIM3v0pTSize,
        .instanceParms = BSIM3v0pTable,
        .numModelParms = &BSIM3v0mPTSize,
        .modelParms = BSIM3v0mPTable,
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

    .DEVparam = BSIM3v0param,
    .DEVmodParam = BSIM3v0mParam,
    .DEVload = BSIM3v0load,
    .DEVsetup = BSIM3v0setup,
    .DEVunsetup = BSIM3v0unsetup,
    .DEVpzSetup = BSIM3v0setup,
    .DEVtemperature = BSIM3v0temp,
    .DEVtrunc = BSIM3v0trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3v0acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3v0getic,
    .DEVask = BSIM3v0ask,
    .DEVmodAsk = BSIM3v0mAsk,
    .DEVpzLoad = BSIM3v0pzLoad,
    .DEVconvTest = BSIM3v0convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3v0noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &BSIM3v0iSize,
    .DEVmodSize = &BSIM3v0mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3v0_info(void)
{
     return &B3v0info; 
}
