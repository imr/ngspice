#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "ndevitf.h"
#include "ndevext.h"
#include "ndevinit.h"


SPICEdev NDEVinfo = {
    {
	"NDEV",
        "Numerical Device",

        &NDEVnSize, /* number of terminals */ 
        &NDEVnSize,
        NDEVnames,  /* the name of terminals*/

        &NDEVpTSize, /*number of instance parameters */
        NDEVpTable,  /*the array of instance parameters */

        &NDEVmPTSize, /* number of model parameter, NDEV does not have this parameter */
        NDEVmPTable,  /*the array of model parameters */
	
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

 /* DEVparam      */ NDEVparam,
 /* DEVmodParam   */ NDEVmParam,
 /* DEVload       */ NDEVload,
 /* DEVsetup      */ NDEVsetup,
 /* DEVunsetup    */ NULL,
 /* DEVpzSetup    */ NDEVsetup,
 /* DEVtemperature*/ NDEVtemp,
 /* DEVtrunc      */ NDEVtrunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ NDEVacLoad,
 /* DEVaccept     */ NDEVaccept,
 /* DEVdestroy    */ NDEVdestroy,
 /* DEVmodDelete  */ NDEVmDelete,
 /* DEVdelete     */ NDEVdelete,
 /* DEVsetic      */ NDEVgetic,
 /* DEVask        */ NDEVask,
 /* DEVmodAsk     */ NULL,
 /* DEVpzLoad     */ NDEVpzLoad,
 /* DEVconvTest   */ NDEVconvTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ NULL,
 /* DEVsoaCheck   */ NULL,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif               
                    
 /* DEVinstSize   */ &NDEViSize,
 /* DEVmodSize    */ &NDEVmSize,

#ifdef KLU
 /* DEVbindCSC        */ NULL,
 /* DEVbindCSCComplex */ NULL,
 /* DEVbindCSCComplexToReal */  NULL,
#endif

};


SPICEdev *
get_ndev_info(void)
{
    return &NDEVinfo;
}
