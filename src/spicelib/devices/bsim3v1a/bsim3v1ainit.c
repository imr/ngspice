#include <config.h>

#include <devdefs.h>

#include "bsim3v1aitf.h"
#include "bsim3v1aext.h"
#include "bsim3v1ainit.h"


SPICEdev B3v1Ainfo = {
    {
        "BSIM3v1A",
        "Berkeley Short Channel IGFET Model Version-3 (3.1 Alan)",

        &BSIM3v1AnSize,
        &BSIM3v1AnSize,
        BSIM3v1Anames,

        &BSIM3v1ApTSize,
        BSIM3v1ApTable,

        &BSIM3v1AmPTSize,
        BSIM3v1AmPTable,
	
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
	
	DEV_DEFAULT,

    },

    DEVparam      : BSIM3v1Aparam,
    DEVmodParam   : BSIM3v1AmParam,
    DEVload       : BSIM3v1Aload,
    DEVsetup      : BSIM3v1Asetup,
    DEVunsetup    : BSIM3v1Aunsetup,
    DEVpzSetup    : BSIM3v1Asetup,
    DEVtemperature: BSIM3v1Atemp,
    DEVtrunc      : BSIM3v1Atrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3v1AacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3v1Adestroy,
    DEVmodDelete  : BSIM3v1AmDelete,
    DEVdelete     : BSIM3v1Adelete, 
    DEVsetic      : BSIM3v1Agetic,
    DEVask        : BSIM3v1Aask,
    DEVmodAsk     : BSIM3v1AmAsk,
    DEVpzLoad     : BSIM3v1ApzLoad,
    DEVconvTest   : BSIM3v1AconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM3v1Anoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                    
    DEVinstSize   : &BSIM3v1AiSize,
    DEVmodSize    : &BSIM3v1AmSize

};


SPICEdev *
get_bsim3v1a_info(void)
{
    return &B3v1Ainfo;
}
