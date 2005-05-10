#include <config.h>

#include <devdefs.h>

#include "numditf.h"
#include "numdext.h"
#include "numdinit.h"


SPICEdev NUMDinfo = {
    {
	"NUMD",
        "1D Numerical Junction Diode model",

        &NUMDnSize,
        &NUMDnSize,
        NUMDnames,

        &NUMDpTSize,
        NUMDpTable,

        &NUMDmPTSize,
        NUMDmPTable,
	
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

    DEVparam      : NUMDparam,
    DEVmodParam   : NUMDmParam,
    DEVload       : NUMDload,
    DEVsetup      : NUMDsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : NUMDsetup,
    DEVtemperature: NUMDtemp,
    DEVtrunc      : NUMDtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NUMDacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : NUMDdestroy,
    DEVmodDelete  : NUMDmDelete,
    DEVdelete     : NUMDdelete,
    DEVsetic      : NULL,
    DEVask        : NUMDask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NUMDpzLoad,
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
    DEVdump	  : NUMDdump,
    DEVacct       : NUMDacct,
#endif
                    
    DEVinstSize   : &NUMDiSize,
    DEVmodSize    : &NUMDmSize

};


SPICEdev *
get_numd_info(void)
{
    return &NUMDinfo;
}
