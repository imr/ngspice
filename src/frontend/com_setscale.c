#include <ngspice.h>
#include <bool.h>
#include <dvec.h>

#include "com_setscale.h"
#include "quote.h"
#include "streams.h"
#include "vectors.h"
#include "plotting/plotting.h"
#include "plotting/pvec.h"


/* Set the default scale to the named vector.  If no vector named,
 * find and print the default scale.  */
void
com_setscale(wordlist *wl)
{
    struct dvec *d;
    char *s;

    if (plot_cur) {
	if (wl) {
	    s = cp_unquote(wl->wl_word);
	    d = vec_get(s);
            if(s) tfree(s);/*DG to avoid the cp_unquote memory leak */
	    if (d == NULL)
		fprintf(cp_err, "Error: no such vector as %s.\n", 
			wl->wl_word);
	    else
		plot_cur->pl_scale = d;
	} else if (plot_cur->pl_scale) {
	    pvec(plot_cur->pl_scale);
	}
    } else {
	fprintf(cp_err, "Error: no current plot.\n");
    }
}
