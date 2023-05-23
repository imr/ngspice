/*************
* com_shell.c
************/
#include <stdio.h>

#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"

#include "com_shell.h"
#include "streams.h"
#include "ngspice/cpextern.h"


#ifdef _WIN32
#define SHELL "cmd /k"
#else
#define SHELL "/bin/sh"
#endif

/* Fork a shell. */
void
com_shell(wordlist *wl)
{
    int   status;
    char *shell = NULL;

    shell = getenv("SHELL");
    if (shell == NULL) {
        shell = SHELL;
    }

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
            char * const com = wl_flatten(wl);
            execl("/bin/sh", "sh", "-c", com, 0);
            txfree(com);
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
        char * const com = wl_flatten(wl);

        status = system(com);
        if (status == -1) {
            (void) fprintf(cp_err, "Unable to execute \"%s\".\n", com);
        }
        txfree(com);
    }
    else {
        status = system(shell);
        if (status == -1) {
            (void) fprintf(cp_err, "Unable to execute \"%s\".\n", shell);
        }
    }
    cp_vset("shellstatus", CP_NUM, &status);
#endif

} /* end of function com_shell */



