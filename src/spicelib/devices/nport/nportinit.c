/**********
Enhancement-242: native n-port device -- SPICEdev descriptor / init.
**********/

#include "ngspice/config.h"
#include <string.h>
#include <stdio.h>
#include "ngspice/devdefs.h"
#include "nportdefs.h"
#include "nportitf.h"
#include "nportinit.h"

/* Fixed maximum terminal count accepted through the N dispatcher.  Sizing the
 * generic GENnode array; an instance uses only its N+1 (ports + ref) nodes. */
static int NPORTnTerms = NPORT_MAXTERMS;

/* generic terminal names ("1".."NPORT_MAXTERMS"), built once on first request */
static char *NPORTnames[NPORT_MAXTERMS];

SPICEdev NPORTinfo = {
    .DEVpublic = {
        .name        = "nport",
        .description = "native n-port rational-model device",
        .terms       = &NPORTnTerms,
        .numNames    = &NPORTnTerms,
        .termNames   = NPORTnames,
        .numInstanceParms = &NPORTpTSize,
        .instanceParms    = NPORTpTable,
        .numModelParms    = &NPORTmPTSize,
        .modelParms       = NPORTmPTable,
        .flags = 0,

#ifdef XSPICE
        .cm_func   = NULL,
        .num_conn  = 0,
        .conn      = NULL,
        .num_param = 0,
        .param     = NULL,
        .num_inst_var = 0,
        .inst_var  = NULL,
#endif
    },

    .DEVparam       = NPORTparam,
    .DEVmodParam    = NPORTmParam,
    .DEVload        = NPORTload,
    .DEVsetup       = NPORTsetup,
    .DEVunsetup     = NPORTunsetup,
    .DEVpzSetup     = NPORTsetup,
    .DEVtemperature = NPORTtemp,
    .DEVtrunc       = NULL,
    .DEVfindBranch  = NULL,
    .DEVacLoad      = NPORTacLoad,
    .DEVaccept      = NULL,
    .DEVdestroy     = NULL,
    .DEVmodDelete   = NULL,
    .DEVdelete      = NPORTdelete,
    .DEVsetic       = NULL,
    .DEVask         = NPORTask,
    .DEVmodAsk      = NPORTmAsk,
    .DEVpzLoad      = NULL,
    .DEVconvTest    = NULL,
    .DEVsenSetup    = NULL,
    .DEVsenLoad     = NULL,
    .DEVsenUpdate   = NULL,
    .DEVsenAcLoad   = NULL,
    .DEVsenPrint    = NULL,
    .DEVsenTrunc    = NULL,
    .DEVdisto       = NULL,
    .DEVnoise       = NULL,
    .DEVsoaCheck    = NULL,

    .DEVinstSize    = &NPORTiSize,
    .DEVmodSize     = &NPORTmSize,

    .DEVbindCSC              = NPORTbindCSC,
    .DEVbindCSCComplex       = NPORTbindCSCComplex,
    .DEVbindCSCComplexToReal = NPORTbindCSCComplexToReal,
};

SPICEdev *
get_nport_info(void)
{
    static int built = 0;
    if (!built) {
        int i;
        for (i = 0; i < NPORT_MAXTERMS; i++) {
            char b[16];
            snprintf(b, sizeof b, "%d", i + 1);
            NPORTnames[i] = strdup(b);
        }
        built = 1;
    }
    return &NPORTinfo;
}
