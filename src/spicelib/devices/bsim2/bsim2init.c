#include <config.h>

#include <devdefs.h>

#include "bsim2itf.h"
#include "bsim2ext.h"
#include "bsim2init.h"


SPICEdev B2info = {
    {
        "BSIM2",
        "Berkeley Short Channel IGFET Model",

        &B2nSize,
        &B2nSize,
        B2names,

        &B2pTSize,
        B2pTable,

        &B2mPTSize,
        B2mPTable,

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

    DEVparam      : B2param,
    DEVmodParam   : B2mParam,
    DEVload       : B2load,
    DEVsetup      : B2setup,
    DEVunsetup    : B2unsetup,
    DEVpzSetup    : B2setup,
    DEVtemperature: B2temp,
    DEVtrunc      : B2trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : B2acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : B2destroy,
    DEVmodDelete  : B2mDelete,
    DEVdelete     : B2delete, 
    DEVsetic      : B2getic,
    DEVask        : B2ask,
    DEVmodAsk     : B2mAsk,
    DEVpzLoad     : B2pzLoad,
    DEVconvTest   : B2convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : B2noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize   : &B2iSize,
    DEVmodSize    : &B2mSize

};


SPICEdev *
get_bsim2_info(void)
{
    return &B2info;
}
