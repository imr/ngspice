#include <config.h>
#include <ngspice.h>

#include <wordlist.h>
#include <cpextern.h>


void
com_rehash(wordlist *wl)
{
    char *s;

    if (!cp_dounixcom) {
        fprintf(cp_err, "Error: unixcom not set.\n");
        return;
    }
    s = getenv("PATH");
    if (s)
        cp_rehash(s, TRUE);
    else
        fprintf(cp_err, "Error: no PATH in environment.\n");
    return;
}

