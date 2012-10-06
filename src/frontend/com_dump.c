#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/inpdefs.h"
#include "circuits.h"
#include "com_dump.h"
#include "ngspice/cpextern.h"
#include "ngspice/fteext.h"
#include "spiceif.h"


void
com_dump(wordlist *wl)
{
    NG_IGNORE(wl);

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }

    if_dump(ft_curckt->ci_ckt, cp_out);
}
