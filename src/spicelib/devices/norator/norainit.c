/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "noraitf.h"
#include "noraext.h"
#include "norainit.h"


SPICEdev NORAinfo = {
    .DEVpublic = {
        .name = "Norator",
        .description = "Norator - arbitrary current source",
        .terms = &NORAnSize,
        .numNames = &NORAnSize,
        .termNames = NORAnames,
        .numInstanceParms = &NORApTSize,
        .instanceParms = NORApTable,
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

    .DEVparam = NORAparam,
    .DEVmodParam = NULL,
    .DEVload = NORAload,
    .DEVsetup = NORAsetup,
    .DEVunsetup = NORAunsetup,
    .DEVpzSetup = NORAsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NORAfindBr,
    .DEVacLoad = NORAload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = NORAask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NORApzLoad,
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
    .DEVinstSize = &NORAiSize,
    .DEVmodSize = &NORAmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_nora_info(void)
{
    return &NORAinfo;
}
