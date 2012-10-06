#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/ftedefs.h"
#include "ngspice/inpdefs.h"

#include "circuits.h"
#include "com_state.h"
#include "ngspice/cpextern.h"
#include "plotting/plotting.h"


void
com_state(wordlist *wl)
{
    NG_IGNORE(wl);

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }
    fprintf(cp_out, "Current circuit: %s\n", ft_curckt->ci_name);
    if (!ft_curckt->ci_inprogress) {
        fprintf(cp_out, "No run in progress.\n");
        return;
    }
    fprintf(cp_out, "Type of run: %s\n", plot_cur->pl_name);
    fprintf(cp_out, "Number of points so far: %d\n",
            plot_cur->pl_scale->v_length);
    fprintf(cp_out, "(That's all this command does so far)\n");
}
