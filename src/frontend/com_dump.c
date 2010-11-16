#include <config.h>
#include <ngspice.h>
#include <bool.h>
#include <wordlist.h>
#include <inpdefs.h>
#include "circuits.h"
#include "com_dump.h"
#include "cpextern.h"
#include "fteext.h"
#include "spiceif.h"


void
com_dump(wordlist *wl)
{
    IGNORE(wl);

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }
    if_dump(ft_curckt->ci_ckt, cp_out);
    return;
}
