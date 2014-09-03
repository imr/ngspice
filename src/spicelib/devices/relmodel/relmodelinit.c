/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "relmodelitf.h"
#include "relmodelext.h"
#include "relmodelinit.h"


SPICEdev RELMODELinfo = {
    {
        "RELMODEL",
        "MOSFET Reliability Analysis Addiction Model",

        &RELMODELnSize,
        &RELMODELnSize,
        RELMODELnames,

        &RELMODELpTSize,
        RELMODELpTable,

        &RELMODELmPTSize,
        RELMODELmPTable,

        DEV_DEFAULT
    },

    NULL,           /* DEVparam       */
    RELMODELmParam, /* DEVmodParam    */
    NULL,           /* DEVload        */
    NULL,           /* DEVsetup       */
    NULL,           /* DEVunsetup     */
    NULL,           /* DEVpzSetup     */
    NULL,           /* DEVtemperature */
    NULL,           /* DEVtrunc       */
    NULL,           /* DEVfindBranch  */
    NULL,           /* DEVacLoad      */
    NULL,           /* DEVaccept      */
    NULL,           /* DEVdestroy     */
    NULL,           /* DEVmodDelete   */
    NULL,           /* DEVdelete      */
    NULL,           /* DEVsetic       */
    NULL,           /* DEVask         */
    RELMODELmAsk,   /* DEVmodAsk      */
    NULL,           /* DEVpzLoad      */
    NULL,           /* DEVconvTest    */
    NULL,           /* DEVsenSetup    */
    NULL,           /* DEVsenLoad     */
    NULL,           /* DEVsenUpdate   */
    NULL,           /* DEVsenAcLoad   */
    NULL,           /* DEVsenPrint    */
    NULL,           /* DEVsenTrunc    */
    NULL,           /* DEVdisto       */
    NULL,           /* DEVnoise       */
    NULL,           /* DEVsoaCheck    */
    &RELMODELiSize, /* DEVinstSize    */
    &RELMODELmSize, /* DEVmodSize     */
    NULL,           /* DEVagingAdd    */
    NULL            /* DEVagingSetup  */
} ;


SPICEdev *
get_relmodel_info (void)
{
    return &RELMODELinfo ;
}
