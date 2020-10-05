/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "nulaitf.h"
#include "nulaext.h"
#include "nulainit.h"


SPICEdev NULAinfo = {
    .DEVpublic = {
        .name = "Nullator",
        .description = "Nullator",
        .terms = &NULAnSize,
        .numNames = &NULAnSize,
        .termNames = NULAnames,
        .numInstanceParms = &NULApTSize,
        .instanceParms = NULApTable,
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

    .DEVparam = NULAparam,
    .DEVmodParam = NULL,
    .DEVload = NULAload,
    .DEVsetup = NULAsetup,
    .DEVunsetup = NULAunsetup,
    .DEVpzSetup = NULAsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = NULAload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = NULAask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NULApzLoad,
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
    .DEVinstSize = &NULAiSize,
    .DEVmodSize = &NULAmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_nula_info(void)
{
    return &NULAinfo;
}
