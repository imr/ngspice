#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"
#include "ngspice/plot.h"
#include "ngspice/fteext.h"

#include "pvec.h"
#include "dimens.h"


void
pvec(struct dvec *d)
{
    char buf[BSIZE_SP], buf2[BSIZE_SP], buf3[BSIZE_SP];

    sprintf(buf, "    %-20s: %s, %s, %d long",
            d->v_name,
            ft_typenames(d->v_type),
            isreal(d) ? "real" : "complex",
            d->v_length);

    if (d->v_flags & VF_MINGIVEN) {
        sprintf(buf2, ", min = %g", d->v_minsignal);
        strcat(buf, buf2);
    }

    if (d->v_flags & VF_MAXGIVEN) {
        sprintf(buf2, ", max = %g", d->v_maxsignal);
        strcat(buf, buf2);
    }

    switch (d->v_gridtype) {
    case GRID_LOGLOG:
        strcat(buf, ", grid = loglog");
        break;

    case GRID_XLOG:
        strcat(buf, ", grid = xlog");
        break;

    case GRID_YLOG:
        strcat(buf, ", grid = ylog");
        break;

    case GRID_POLAR:
        strcat(buf, ", grid = polar");
        break;

    case GRID_SMITH:
        strcat(buf, ", grid = smith (xformed)");
        break;

    case GRID_SMITHGRID:
        strcat(buf, ", grid = smithgrid (not xformed)");
        break;

    default: /* va: GRID_NONE or GRID_LIN */
        break;
    }

    switch (d->v_plottype) {

    case PLOT_COMB:
        strcat(buf, ", plot = comb");
        break;

    case PLOT_POINT:
        strcat(buf, ", plot = point");
        break;

    default:  /* va: PLOT_LIN, */
        break;
    }

    if (d->v_defcolor) {
        sprintf(buf2, ", color = %s", d->v_defcolor);
        strcat(buf, buf2);
    }

    if (d->v_scale) {
        sprintf(buf2, ", scale = %s", d->v_scale->v_name);
        strcat(buf, buf2);
    }

    if (d->v_numdims > 1) {
        dimstring(d->v_dims, d->v_numdims, buf3);
        size_t icopy = BSIZE_SP - 1;
        size_t len = (size_t)snprintf(buf2, icopy, ", dims = [%s]", buf3);
        if (len > icopy) {
            fprintf(stderr, "Warning: Potential buffer overflow while setting a vector dimension");
        }
        strcat(buf, buf2);
    }

    if (d->v_plot->pl_scale == d)
        strcat(buf, " [default scale]\n");
    else
        strcat(buf, "\n");

    out_send(buf);
}
