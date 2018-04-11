#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "soi3itf.h"
#include "soi3ext.h"
#include "soi3init.h"


SPICEdev SOI3info = {
    .DEVpublic = {
        .name = "SOI3",
        .description = "Basic Thick Film SOI3 model v2.7",
        .terms = &SOI3nSize,
        .numNames = &SOI3nSize,
        .termNames = SOI3names,
        .numInstanceParms = &SOI3pTSize,
        .instanceParms = SOI3pTable,
        .numModelParms = &SOI3mPTSize,
        .modelParms = SOI3mPTable,
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

    .DEVparam = SOI3param,
    .DEVmodParam = SOI3mParam,
    .DEVload = SOI3load,
    .DEVsetup = SOI3setup,
    .DEVunsetup = SOI3unsetup,
    .DEVpzSetup = SOI3setup,
    .DEVtemperature = SOI3temp,
    .DEVtrunc = SOI3trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = SOI3acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = SOI3getic,
    .DEVask = SOI3ask,
    .DEVmodAsk = SOI3mAsk,
    .DEVpzLoad = NULL,
    .DEVconvTest = SOI3convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = SOI3noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &SOI3iSize,
    .DEVmodSize = &SOI3mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_soi3_info(void)
{
    return &SOI3info;
}
