#include <config.h>

#include <devdefs.h>

#include "nbjtitf.h"
#include "nbjtext.h"
#include "nbjtinit.h"


SPICEdev NBJTinfo = {
    {
	"NBJT",
        "1D Numerical Bipolar Junction Transistor model",

        &NBJTnSize,
        &NBJTnSize,
        NBJTnames,

        &NBJTpTSize,
        NBJTpTable,

        &NBJTmPTSize,
        NBJTmPTable,

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

    DEVparam      : NBJTparam,
    DEVmodParam   : NBJTmParam,
    DEVload       : NBJTload,
    DEVsetup      : NBJTsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : NBJTsetup,
    DEVtemperature: NBJTtemp,
    DEVtrunc      : NBJTtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NBJTacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : NBJTdestroy,
    DEVmodDelete  : NBJTmDelete,
    DEVdelete     : NBJTdelete,
    DEVsetic      : NULL,
    DEVask        : NBJTask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NBJTpzLoad,
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
    DEVdump       : NBJTdump,
    DEVacct       : NBJTacct,
#endif

    DEVinstSize   : &NBJTiSize,
    DEVmodSize    : &NBJTmSize

};


SPICEdev *
get_nbjt_info(void)
{
    return &NBJTinfo;
}
