#include <config.h>

#include <devdefs.h>

#include "mos2itf.h"
#include "mos2ext.h"
#include "mos2init.h"


SPICEdev MOS2info = {
    {
        "Mos2",
        "Level 2 MOSfet model with Meyer capacitance model",

        &MOS2nSize,
        &MOS2nSize,
        MOS2names,

        &MOS2pTSize,
        MOS2pTable,

        &MOS2mPTSize,
        MOS2mPTable,

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

    DEVparam      : MOS2param,
    DEVmodParam   : MOS2mParam,
    DEVload       : MOS2load,
    DEVsetup      : MOS2setup,
    DEVunsetup    : MOS2unsetup,
    DEVpzSetup    : MOS2setup,
    DEVtemperature: MOS2temp,
    DEVtrunc      : MOS2trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MOS2acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MOS2destroy,
    DEVmodDelete  : MOS2mDelete,
    DEVdelete     : MOS2delete,
    DEVsetic      : MOS2getic,
    DEVask        : MOS2ask,
    DEVmodAsk     : MOS2mAsk,
    DEVpzLoad     : MOS2pzLoad,
    DEVconvTest   : MOS2convTest,
    DEVsenSetup   : MOS2sSetup,
    DEVsenLoad    : MOS2sLoad,
    DEVsenUpdate  : MOS2sUpdate,
    DEVsenAcLoad  : MOS2sAcLoad,
    DEVsenPrint   : MOS2sPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : MOS2disto,
    DEVnoise      : MOS2noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize   : &MOS2iSize,
    DEVmodSize    : &MOS2mSize
};


SPICEdev *
get_mos2_info(void)
{
    return &MOS2info;
}
