#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vsrcitf.h"
#include "vsrcext.h"
#include "vsrcinit.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

SPICEdev VSRCinfo = {
    {
        "Vsource", 
        "Independent voltage source",

        &VSRCnSize,
        &VSRCnSize,
        VSRCnames,

        &VSRCpTSize,
        VSRCpTable,

        NULL,
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

 /* DEVparam      */ VSRCparam,
 /* DEVmodParam   */ NULL,
#ifdef USE_CUSPICE
 /* DEVload       */ cuVSRCload,
#else
 /* DEVload       */ VSRCload,
#endif
 /* DEVsetup      */ VSRCsetup,
 /* DEVunsetup    */ VSRCunsetup,
 /* DEVpzSetup    */ VSRCpzSetup,
 /* DEVtemperature*/ VSRCtemp,
 /* DEVtrunc      */ NULL,
 /* DEVfindBranch */ VSRCfindBr,
 /* DEVacLoad     */ VSRCacLoad,
 /* DEVaccept     */ VSRCaccept,
 /* DEVdestroy    */ VSRCdestroy,
 /* DEVmodDelete  */ VSRCmDelete,
 /* DEVdelete     */ VSRCdelete,
 /* DEVsetic      */ NULL,
 /* DEVask        */ VSRCask,
 /* DEVmodAsk     */ NULL,
 /* DEVpzLoad     */ VSRCpzLoad,
 /* DEVconvTest   */ NULL,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL, /* DISTO */
 /* DEVnoise      */ NULL, /* NOISE */
 /* DEVsoaCheck   */ NULL,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif                        
 /* DEVinstSize   */ &VSRCiSize,
 /* DEVmodSize    */ &VSRCmSize,

#ifdef KLU
 /* DEVbindCSC        */ VSRCbindCSC,
 /* DEVbindCSCComplex */ VSRCbindCSCComplex,
 /* DEVbindCSCComplexToReal */  VSRCbindCSCComplexToReal,
#endif

#ifdef USE_CUSPICE
 /* cuDEVdestroy */ cuVSRCdestroy,
 /* DEVtopology */  VSRCtopology,
#endif

};


SPICEdev *
get_vsrc_info(void)
{
    return &VSRCinfo;
}
