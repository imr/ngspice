#include <config.h>

#include <devdefs.h>

#include "asrcitf.h"
#include "asrcext.h"
#include "asrcinit.h"


SPICEdev ASRCinfo = {
    {
	"ASRC",
	"Arbitrary Source ",

	&ASRCnSize,
	&ASRCnSize,
	ASRCnames,

	&ASRCpTSize,
	ASRCpTable,

	0,
	NULL,
	
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

    DEVparam      : ASRCparam,
    DEVmodParam   : NULL,
    DEVload       : ASRCload,
    DEVsetup      : ASRCsetup,
    DEVunsetup    : ASRCunsetup,
    DEVpzSetup    : ASRCsetup,
    DEVtemperature: NULL,
    DEVtrunc      : NULL,
    DEVfindBranch : ASRCfindBr,
    DEVacLoad     : ASRCacLoad,   /* ac and normal load functions NOT identical */
    DEVaccept     : NULL,
    DEVdestroy    : ASRCdestroy,
    DEVmodDelete  : ASRCmDelete,
    DEVdelete     : ASRCdelete,
    DEVsetic      : NULL,
    DEVask        : ASRCask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : ASRCpzLoad,
    DEVconvTest   : ASRCconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump	  : NULL,
    DEVacct       : NULL,
#endif                    
    DEVinstSize   : &ASRCiSize,
    DEVmodSize    : &ASRCmSize
};


SPICEdev *
get_asrc_info(void)
{
    return &ASRCinfo;
}
