#include <config.h>

#include <devdefs.h>

#include "mesaitf.h"
#include "mesaext.h"
#include "mesainit.h"


SPICEdev MESAinfo = {
    {
        "MESA",
        "GaAs MESFET model",

        &MESAnSize,
        &MESAnSize,
        MESAnames,

        &MESApTSize,
        MESApTable,

        &MESAmPTSize,
        MESAmPTable,

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

    DEVparam      : MESAparam,
    DEVmodParam   : MESAmParam,
    DEVload       : MESAload,
    DEVsetup      : MESAsetup,
    DEVunsetup    : MESAunsetup,
    DEVpzSetup    : MESAsetup,
    DEVtemperature: MESAtemp,
    DEVtrunc      : MESAtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MESAacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MESAdestroy,
    DEVmodDelete  : MESAmDelete,
    DEVdelete     : MESAdelete,
    DEVsetic      : MESAgetic,
    DEVask        : MESAask,
    DEVmodAsk     : MESAmAsk,
    DEVpzLoad     : MESApzLoad,
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
    DEVinstSize   : &MESAiSize,
    DEVmodSize    : &MESAmSize

};


SPICEdev *
get_mesa_info(void)
{
    return &MESAinfo;
}
