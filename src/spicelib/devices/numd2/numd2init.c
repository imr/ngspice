#include <config.h>

#include <devdefs.h>

#include "numd2itf.h"
#include "numd2ext.h"
#include "numd2init.h"


SPICEdev NUMD2info = {
    {
	"NUMD2",
        "2D Numerical Junction Diode model",

        &NUMD2nSize,
        &NUMD2nSize,
        NUMD2names,

        &NUMD2pTSize,
        NUMD2pTable,

        &NUMD2mPTSize,
        NUMD2mPTable,
	
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

    DEVparam      : NUMD2param,
    DEVmodParam   : NUMD2mParam,
    DEVload       : NUMD2load,
    DEVsetup      : NUMD2setup,
    DEVunsetup    : NULL,
    DEVpzSetup    : NUMD2setup,
    DEVtemperature: NUMD2temp,
    DEVtrunc      : NUMD2trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NUMD2acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : NUMD2destroy,
    DEVmodDelete  : NUMD2mDelete,
    DEVdelete     : NUMD2delete,
    DEVsetic      : NULL,
    DEVask        : NUMD2ask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NUMD2pzLoad,
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
    DEVdump       : NUMD2dump,
    DEVacct       : NUMD2acct,
#endif
                    
    DEVinstSize   : &NUMD2iSize,
    DEVmodSize    : &NUMD2mSize

};


SPICEdev *
get_numd2_info(void)
{
    return &NUMD2info;
}
