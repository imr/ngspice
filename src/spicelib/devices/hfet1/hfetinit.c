#include <config.h>

#include <devdefs.h>

#include "hfetitf.h"
#include "hfetext.h"
#include "hfetinit.h"


SPICEdev HFETAinfo = {
    {
        "HFET1",
        "HFET1 Model",

        &HFETAnSize,
        &HFETAnSize,
        HFETAnames,

        &HFETApTSize,
        HFETApTable,

        &HFETAmPTSize,
        HFETAmPTable,

#ifdef XSPICE
/*----  Fixed by SDB 5.2.2003 to enable XSPICE/tclspice integration  -----*/
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */
/*---------------------------  End of SDB fix   -------------------------*/
#endif

	DEV_DEFAULT
    },

    DEVparam      : HFETAparam,
    DEVmodParam   : HFETAmParam,
    DEVload       : HFETAload,
    DEVsetup      : HFETAsetup,
    DEVunsetup    : HFETAunsetup,
    DEVpzSetup    : HFETAsetup,
    DEVtemperature: HFETAtemp,
    DEVtrunc      : HFETAtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : HFETAacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : HFETAdestroy,
    DEVmodDelete  : HFETAmDelete,
    DEVdelete     : HFETAdelete,
    DEVsetic      : HFETAgetic,
    DEVask        : HFETAask,
    DEVmodAsk     : HFETAmAsk,
    DEVpzLoad     : HFETApzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : NULL,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize   : &HFETAiSize,
    DEVmodSize    : &HFETAmSize

};


SPICEdev *
get_hfeta_info(void)
{
    return &HFETAinfo;
}
