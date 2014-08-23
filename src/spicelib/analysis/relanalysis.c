/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"

int
RELANanalysis (CKTcircuit *ckt, int notused)
{
    int converged ;
    int error ;
    IFuid *nameList ;
    int numNames ;
    runDesc *plot = NULL ;

    NG_IGNORE (notused) ;
  
    error = CKTnames (ckt, &numNames, &nameList) ;
    if (error)
    {
        return (error) ;
    }
    error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob, ckt->CKTcurJob->JOBname, NULL, IF_REAL,
                                       numNames, nameList, IF_REAL, &plot) ;
    tfree (nameList) ;
    if (error)
    {
        return (error) ;
    }

    /* initialize CKTsoaCheck `warn' counters */
    if (ckt->CKTsoaCheck)
    {
        error = CKTsoaInit () ;
    }

    converged = CKTop (ckt, (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                       (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT, ckt->CKTdcMaxIter) ;
    if (converged != 0)
    {
        fprintf (stdout, "\nDC solution failed -\n") ;
        CKTncDump (ckt) ;	   
        return (converged) ;
    }

    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG ;

    converged = CKTload (ckt) ;
    if (converged == 0)
    {
        CKTdump (ckt, 0.0, plot) ;
        if (ckt->CKTsoaCheck)
        {
            error = CKTsoaCheck (ckt) ;
        }
    } else {
        fprintf (stderr, "error: circuit reload failed.\n") ;
    }

    SPfrontEnd->OUTendPlot (plot) ;


    /* Extract Vth and calculate delvto for every instance */
    error = CKTagingSetup (ckt) ;


    return (error) ;
}
