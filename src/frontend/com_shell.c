/*************
* com_shell.c
************/

#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"

#include "com_shell.h"
#include "streams.h"
#include "ngspice/cpextern.h"


/* Fork a shell. */

void
com_shell(wordlist *wl)
{
    char *com, *shell = NULL;

    shell = getenv("SHELL");
    if (shell == NULL)
        shell = "/bin/csh";

    cp_ccon(FALSE);

#ifdef HAVE_VFORK_H
    /* XXX Needs to switch process groups.  Also, worry about suspend */
    /* Only bother for efficiency */
    pid = vfork();
    if (pid == 0) {
        fixdescriptors();
        if (wl == NULL) {
            execl(shell, shell, 0);
            _exit(99);
        } else {
            com = wl_flatten(wl);
            execl("/bin/sh", "sh", "-c", com, 0);
        }
    } else {
        /* XXX Better have all these signals */
        svint = signal(SIGINT, SIG_DFL);
        svquit = signal(SIGQUIT, SIG_DFL);
        svtstp = signal(SIGTSTP, SIG_DFL);
        /* XXX Sig on proc group */
        do
            r = wait(NULL);
        while ((r != pid) && pid != -1);
        signal(SIGINT, (SIGNAL_FUNCTION) svint);
        signal(SIGQUIT, (SIGNAL_FUNCTION) svquit);
        signal(SIGTSTP, (SIGNAL_FUNCTION) svtstp);
    }
#else
    /* Easier to forget about changing the io descriptors. */
    if (wl) {
        com = wl_flatten(wl);
        system(com);
        tfree(com);
    } else {
        system(shell);
    }
#endif

}
