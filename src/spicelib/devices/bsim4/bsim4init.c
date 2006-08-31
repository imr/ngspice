#include "config.h"

#include "devdefs.h"

#include "bsim4itf.h"
#include "bsim4ext.h"
#include "bsim4init.h"


SPICEdev BSIM4info = {
    {
        "BSIM4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4nSize,
        &BSIM4nSize,
        BSIM4names,

        &BSIM4pTSize,
        BSIM4pTable,

        &BSIM4mPTSize,
        BSIM4mPTable,

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

    BSIM4param,    /* DEVparam       */
    BSIM4mParam,   /* DEVmodParam    */
    BSIM4load,     /* DEVload        */
    BSIM4setup,    /* DEVsetup       */
    BSIM4unsetup,  /* DEVunsetup     */
    BSIM4setup,    /* DEVpzSetup     */
    BSIM4temp,     /* DEVtemperature */
    BSIM4trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4destroy,  /* DEVdestroy     */
    BSIM4mDelete,  /* DEVmodDelete   */
    BSIM4delete,   /* DEVdelete      */
    BSIM4getic,    /* DEVsetic       */
    BSIM4ask,      /* DEVask         */
    BSIM4mAsk,     /* DEVmodAsk      */
    BSIM4pzLoad,   /* DEVpzLoad      */
    BSIM4convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4noise,    /* DEVnoise       */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4iSize,   /* DEVinstSize    */
    &BSIM4mSize    /* DEVmodSize     */
};


SPICEdev *
get_bsim4_info(void)
{
    return &BSIM4info;
}
