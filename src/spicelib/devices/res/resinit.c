#include <config.h>

#include <devdefs.h>

#include "resitf.h"
#include "resext.h"
#include "resinit.h"


SPICEdev RESinfo = {
    {
        "Resistor",
        "Simple linear resistor",

        &RESnSize,
        &RESnSize,
        RESnames,

        &RESpTSize,
        RESpTable,

        &RESmPTSize,
        RESmPTable,

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

	0
    },

    DEVparam      : RESparam,
    DEVmodParam   : RESmParam,
    DEVload       : RESload,
    DEVsetup      : RESsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : RESsetup,
    DEVtemperature: REStemp,
    DEVtrunc      : NULL,
    DEVfindBranch : NULL,
    DEVacLoad     : RESacload,  /* ac load and normal load are identical */
    DEVaccept     : NULL,
    DEVdestroy    : RESdestroy,
    DEVmodDelete  : RESmDelete,
    DEVdelete     : RESdelete,
    DEVsetic      : NULL,
    DEVask        : RESask,
    DEVmodAsk     : RESmodAsk,
    DEVpzLoad     : RESpzLoad,
    DEVconvTest   : NULL,     /* RESconvTest, XXXX experimental */
    DEVsenSetup   : RESsSetup,
    DEVsenLoad    : RESsLoad,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : RESsAcLoad,
    DEVsenPrint   : RESsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : RESnoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &RESiSize,
    DEVmodSize    : &RESmSize

};


SPICEdev *
get_res_info(void)
{
    return &RESinfo;
}
