#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v0itf.h"
#include "bsim4v0ext.h"
#include "bsim4v0init.h"


SPICEdev BSIM4v0info = {
    {
        "BSIM4v0",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v0nSize,
        &BSIM4v0nSize,
        BSIM4v0names,

        &BSIM4v0pTSize,
        BSIM4v0pTable,

        &BSIM4v0mPTSize,
        BSIM4v0mPTable,

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

    BSIM4v0param,    /* DEVparam       */
    BSIM4v0mParam,   /* DEVmodParam    */
    BSIM4v0load,     /* DEVload        */
    BSIM4v0setup,    /* DEVsetup       */
    BSIM4v0unsetup,  /* DEVunsetup     */
    BSIM4v0setup,    /* DEVpzSetup     */
    BSIM4v0temp,     /* DEVtemperature */
    BSIM4v0trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4v0acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4v0destroy,  /* DEVdestroy     */
    BSIM4v0mDelete,  /* DEVmodDelete   */
    BSIM4v0delete,   /* DEVdelete      */
    BSIM4v0getic,    /* DEVsetic       */
    BSIM4v0ask,      /* DEVask         */
    BSIM4v0mAsk,     /* DEVmodAsk      */
    BSIM4v0pzLoad,   /* DEVpzLoad      */
    BSIM4v0convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4v0noise,    /* DEVnoise       */
    //BSIM4v0soaCheck,/* DEVsoaCheck    */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4v0iSize,   /* DEVinstSize    */
    &BSIM4v0mSize    /* DEVmodSize     */
};


SPICEdev *
get_bsim4v0_info(void)
{
    return &BSIM4v0info;
}
