#include <config.h>

#include <devdefs.h>

#include "bjtitf.h"
#include "bjtext.h"
#include "bjtinit.h"


SPICEdev BJTinfo = {                  /* description from struct IFdevice  */  
    {
      "BJT",                          /*  char *name               */
      "Bipolar Junction Transistor",  /*  char *description        */
      
      &BJTnSize,                      /*  int *terms               */
      &BJTnSize,                      /*  int *numNames            */
      BJTnames,                       /*  char **termnames         */
      
      &BJTpTSize,                     /*  int *numInstanceparms    */
      BJTpTable,                      /*  IFparm *instanceParms    */
      
      &BJTmPTSize,                    /*  int *numModelparms       */
      BJTmPTable,                     /*  IFparm *modelParms       */

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

      DEV_DEFAULT                     /*  int flags                */
    },
    
    DEVparam      : BJTparam,
    DEVmodParam   : BJTmParam,
    DEVload       : BJTload,
    DEVsetup      : BJTsetup,
    DEVunsetup    : BJTunsetup,
    DEVpzSetup    : BJTsetup,
    DEVtemperature: BJTtemp,
    DEVtrunc      : BJTtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BJTacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BJTdestroy,
    DEVmodDelete  : BJTmDelete,
    DEVdelete     : BJTdelete,
    DEVsetic      : BJTgetic,
    DEVask        : BJTask,
    DEVmodAsk     : BJTmAsk,
    DEVpzLoad     : BJTpzLoad,
    DEVconvTest   : BJTconvTest,
    DEVsenSetup   : BJTsSetup,
    DEVsenLoad    : BJTsLoad,
    DEVsenUpdate  : BJTsUpdate,
    DEVsenAcLoad  : BJTsAcLoad,
    DEVsenPrint   : BJTsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : BJTdisto,
    DEVnoise      : BJTnoise,
#ifdef CIDER
    DEVdump	  : NULL,
    DEVacct       : NULL,
#endif                     
    DEVinstSize   : &BJTiSize,
    DEVmodSize    : &BJTmSize

};


SPICEdev *
get_bjt_info(void)
{
    return &BJTinfo;
}
