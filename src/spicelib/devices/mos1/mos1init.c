#include <config.h>

#include <devdefs.h>

#include "mos1itf.h"
#include "mos1ext.h"
#include "mos1init.h"


SPICEdev MOS1info = {
    {
        "Mos1",
        "Level 1 MOSfet model with Meyer capacitance model",

        &MOS1nSize,
        &MOS1nSize,
        MOS1names,

        &MOS1pTSize,
        MOS1pTable,

        &MOS1mPTSize,
        MOS1mPTable,

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

    DEVparam      : MOS1param,
    DEVmodParam   : MOS1mParam,
    DEVload       : MOS1load,
    DEVsetup      : MOS1setup,
    DEVunsetup    : MOS1unsetup,
    DEVpzSetup    : MOS1setup,
    DEVtemperature: MOS1temp,
    DEVtrunc      : MOS1trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MOS1acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MOS1destroy,
    DEVmodDelete  : MOS1mDelete,
    DEVdelete     : MOS1delete,
    DEVsetic      : MOS1getic,
    DEVask        : MOS1ask,
    DEVmodAsk     : MOS1mAsk,
    DEVpzLoad     : MOS1pzLoad,
    DEVconvTest   : MOS1convTest,
    DEVsenSetup   : MOS1sSetup,
    DEVsenLoad    : MOS1sLoad,
    DEVsenUpdate  : MOS1sUpdate,
    DEVsenAcLoad  : MOS1sAcLoad,
    DEVsenPrint   : MOS1sPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : MOS1disto,
    DEVnoise      : MOS1noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                       
    DEVinstSize   : &MOS1iSize,
    DEVmodSize    : &MOS1mSize
};


SPICEdev *
get_mos1_info(void)
{
    return &MOS1info;
}
