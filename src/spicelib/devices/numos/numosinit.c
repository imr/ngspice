#include <config.h>

#include <devdefs.h>

#include "numositf.h"
#include "numosext.h"
#include "numosinit.h"


SPICEdev NUMOSinfo = {
    {
	"NUMOS",
        "2D Numerical MOS Field Effect Transistor model",

        &NUMOSnSize,
        &NUMOSnSize,
        NUMOSnames,

        &NUMOSpTSize,
        NUMOSpTable,

        &NUMOSmPTSize,
        NUMOSmPTable,
	
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

    DEVparam      : NUMOSparam,
    DEVmodParam   : NUMOSmParam,
    DEVload       : NUMOSload,
    DEVsetup      : NUMOSsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : NUMOSsetup,
    DEVtemperature: NUMOStemp,
    DEVtrunc      : NUMOStrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NUMOSacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : NUMOSdestroy,
    DEVmodDelete  : NUMOSmDelete,
    DEVdelete     : NUMOSdelete,
    DEVsetic      : NULL,
    DEVask        : NUMOSask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NUMOSpzLoad,
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
    DEVdump       : NUMOSdump,
    DEVacct       : NUMOSacct,
#endif
                    
    DEVinstSize   : &NUMOSiSize,
    DEVmodSize    : &NUMOSmSize

};


SPICEdev *
get_numos_info(void)
{
    return &NUMOSinfo;
}
