#include "config.h"

#include "devdefs.h"
#include "cktdefs.h"

#include "diodefs.h"
#include "dioitf.h"
#include "dioext.h"
#include "dioinit.h"


SPICEdev DIOinfo = {
    {
        "Diode",
        "Junction Diode model",

        &DIOnSize,
        &DIOnSize,
        DIOnames,

        &DIOpTSize,
        DIOpTable,

        &DIOmPTSize,
        DIOmPTable,

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

    DEVparam      : DIOparam,
    DEVmodParam   : DIOmParam,
    DEVload       : DIOload,
    DEVsetup      : DIOsetup,
    DEVunsetup    : DIOunsetup,
    DEVpzSetup    : DIOsetup,
    DEVtemperature: DIOtemp,
    DEVtrunc      : DIOtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : DIOacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : DIOdestroy,
    DEVmodDelete  : DIOmDelete,
    DEVdelete     : DIOdelete,
    DEVsetic      : DIOgetic,
    DEVask        : DIOask,
    DEVmodAsk     : DIOmAsk,
    DEVpzLoad     : DIOpzLoad,
    DEVconvTest   : DIOconvTest,
    DEVsenSetup   : DIOsSetup,
    DEVsenLoad    : DIOsLoad,
    DEVsenUpdate  : DIOsUpdate,
    DEVsenAcLoad  : DIOsAcLoad,
    DEVsenPrint   : DIOsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : DIOdisto,
    DEVnoise      : DIOnoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                     
    DEVinstSize   : &DIOiSize,
    DEVmodSize    : &DIOmSize
};


SPICEdev *
get_dio_info(void)
{
    return &DIOinfo;
}
