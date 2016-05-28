/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ftehelp.h"
#include "ngspice/hlpdefs.h"
#include "misccoms.h"
#include "postcoms.h"
#include "circuits.h"
#include "variable.h"
#include "plotting/graf.h"
#include "display.h"
#include "../misc/ivars.h"
#include "com_alias.h"
#include "define.h"
#include "runcoms2.h"

#ifdef HAVE_GNUREADLINE
#include <readline/readline.h>
#include <readline/history.h>
extern char history_file[];
#endif

#ifdef HAVE_BSDEDITLINE
#include <editline/readline.h>
extern char history_file[];
#endif

extern IFsimulator SIMinfo;
extern void spice_destroy_devices(void); /* FIXME need a better place */
static void byemesg(void);
static int  confirm_quit(void);


void
com_quit(wordlist *wl)
{
    int exitcode = EXIT_NORMAL;

    bool noask =
        (wl  &&  wl->wl_word  &&  1 == sscanf(wl->wl_word, "%d", &exitcode)) ||
        (wl  &&  wl->wl_word  &&  cieq(wl->wl_word, "noask"))  ||
        !cp_getvar("askquit", CP_BOOL, NULL);

    /* update screen and reset terminal */
    gr_clean();
    cp_ccon(FALSE);

    /* Make sure the guy really wants to quit. */
    if (!ft_nutmeg)
        if (!noask && !confirm_quit())
            return;

#ifndef SHARED_MODULE
	if (!ft_ngdebug)
	    exit(exitcode);
#endif

    /* start to clean up the mess */

#ifdef SHARED_MODULE
    {
        wordlist all = { "all", NULL, NULL };
        wordlist star = { "*", NULL, NULL };

//      com_remcirc(NULL);
        com_destroy(&all);
        com_unalias(&star);
        com_undefine(&star);

        cp_remvar("history");
        cp_remvar("noglob");
        cp_remvar("brief");
        cp_remvar("sourcepath");
        cp_remvar("program");
        cp_remvar("prompt");
    }
#endif

#ifdef EXPERIMENTAL_CODE
    /* Destroy CKT when quit. Add by Gong Ding, gdiso@ustc.edu */
    if (!ft_nutmeg) {
        struct circ *cc;
        for (cc = ft_circuits; cc; cc = cc->ci_next)
            if (SIMinfo.deleteCircuit)
                SIMinfo.deleteCircuit(cc->ci_ckt);
    }
#endif

#ifdef SHARED_MODULE
    /* Destroy CKT when quit. */
    if (!ft_nutmeg) {
        while(ft_curckt)
            com_remcirc(NULL);
    }
#endif

    DevSwitch(NULL);
    DevSwitch(NULL);

    /* then go away */

#ifdef SHARED_MODULE
    cp_destroy_keywords();
    destroy_ivars();
#endif

    byemesg();
#ifdef SHARED_MODULE
    destroy_const_plot();
    spice_destroy_devices();
#endif
    mc_free();
#ifdef SHARED_MODULE
    /* add 1000 to notify that we exit from 'quit' */
    controlled_exit(1000 + exitcode);
#else
    exit(exitcode);
#endif
}


#ifdef SYSTEM_MAIL

void
com_bug(wordlist *wl)
{
    char buf[BSIZE_SP];

    NG_IGNORE(wl);

    if (!Bug_Addr || !*Bug_Addr) {
        fprintf(cp_err, "Error: No address to send bug reports to.\n");
        return;
    }

    fprintf(cp_out,
            "Calling the mail program . . .(sending to %s)\n\n"
            "Please include the OS version number and machine architecture.\n"
            "If the problem is with a specific circuit, please include the\n"
            "input file.\n",
            Bug_Addr);

    (void) sprintf(buf, SYSTEM_MAIL, ft_sim->simulator, ft_sim->version, Bug_Addr);
    (void) system(buf);

    fprintf(cp_out, "Bug report sent.  Thank you.\n");
}

#else

void
com_bug(wordlist *wl)
{
    NG_IGNORE(wl);

    fprintf(cp_out,
            "Please use the ngspice bug tracker at:\n"
            "http://sourceforge.net/p/ngspice/bugs/\n");
}

#endif


/* printout upon startup or 'version' command. options to version are -s (short)
   or -f (full). 'version' with options may also be used in ngspice pipe mode. */

void
com_version(wordlist *wl)
{
    if (!wl) {

        /* no printout in pipe mode (-p) */
        if (ft_pipemode)
            return;

        fprintf(cp_out,
                "******\n"
                "** %s-%s : %s\n"
                "** The U. C. Berkeley CAD Group\n"
                "** Copyright 1985-1994, Regents of the University of California.\n"
                "** %s\n",
                ft_sim->simulator, ft_sim->version, ft_sim->description, Spice_Manual);
        if (Spice_Notice != NULL && *Spice_Notice != '\0')
            fprintf(cp_out, "** %s\n", Spice_Notice);
        if (Spice_Build_Date != NULL && *Spice_Build_Date != '\0')
            fprintf(cp_out, "** Creation Date: %s\n", Spice_Build_Date);
        fprintf(cp_out, "******\n");

    } else {

        char *s = wl_flatten(wl);

        if (!strncasecmp(s, "-s", 2)) {

            fprintf(cp_out,
                    "******\n"
                    "** %s-%s\n"
                    "** %s\n",
                    ft_sim->simulator, ft_sim->version, Spice_Manual);
            if (Spice_Notice != NULL && *Spice_Notice != '\0')
                fprintf(cp_out, "** %s\n", Spice_Notice);
            if (Spice_Build_Date != NULL && *Spice_Build_Date != '\0')
                fprintf(cp_out, "** Creation Date: %s\n", Spice_Build_Date);
            fprintf(cp_out, "******\n");

        } else if (!strncasecmp(s, "-f", 2)) {

            fprintf(cp_out,
                    "******\n"
                    "** %s-%s : %s\n"
                    "** The U. C. Berkeley CAD Group\n"
                    "** Copyright 1985-1994, Regents of the University of California.\n"
                    "** %s\n",
                    ft_sim->simulator, ft_sim->version, ft_sim->description, Spice_Manual);
            if (Spice_Notice != NULL && *Spice_Notice != '\0')
                fprintf(cp_out, "** %s\n", Spice_Notice);
            if (Spice_Build_Date != NULL && *Spice_Build_Date != '\0')
                fprintf(cp_out, "** Creation Date: %s\n", Spice_Build_Date);
            fprintf(cp_out, "**\n");
#ifdef CIDER
            fprintf(cp_out, "** CIDER 1.b1 (CODECS simulator) included\n");
#endif
#ifdef XSPICE
            fprintf(cp_out, "** XSPICE extensions included\n");
#endif
            fprintf(cp_out, "** Relevant compilation options (refer to user's manual):\n");
#ifdef NGDEBUG
            fprintf(cp_out, "** Debugging option (-g) enabled\n");
#endif
#ifdef ADMS
            fprintf(cp_out, "** Adms interface enabled\n");
#endif
#ifdef USE_OMP
            fprintf(cp_out, "** OpenMP multithreading for BSIM3, BSIM4 enabled\n");
#endif
#if defined(X_DISPLAY_MISSING) && !defined(HAS_WINGUI)
            fprintf(cp_out, "** X11 interface not compiled into ngspice\n");
#endif
#ifdef NOBYPASS
            fprintf(cp_out, "** --enable-nobypass\n");
#endif
#ifdef CAPBYPASS
            fprintf(cp_out, "** --enable-capbypass\n");
#endif
#ifdef NODELIMITING
            fprintf(cp_out, "** --enable-nodelimiting\n");
#endif
#ifdef PREDICTOR
            fprintf(cp_out, "** --enable-predictor\n");
#endif
#ifdef NEWTRUNC
            fprintf(cp_out, "** --enable-newtrunc\n");
#endif
#ifdef WANT_SENSE2
            fprintf(cp_out, "** --enable-sense2\n");
#endif
            fprintf(cp_out, "**\n");
#ifdef EXPERIMENTAL_CODE
            fprintf(cp_out, "** Experimental code enabled.\n");
#endif
#ifdef EXP_DEV
            fprintf(cp_out, "** Experimental devices enabled.\n");
#endif
            fprintf(cp_out, "******\n");

        } else if (!eq(ft_sim->version, s)) {

            fprintf(stderr,
                    "Note: rawfile is version %s (current version is %s)\n",
                    wl->wl_word, ft_sim->version);

        }

        tfree(s);
    }
}


static void
byemesg(void)
{

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    /*  write out command history only when saying goodbye.  */
    if (cp_interactive && (cp_maxhistlength > 0)) {
        stifle_history(cp_maxhistlength);
        write_history(history_file);
    }
#endif

    printf("%s-%s done\n", ft_sim->simulator, ft_sim->version);
}


static int
confirm_quit(void)
{
    struct circ *cc;
    struct plot *pl;
    int ncc = 0, npl = 0;
    char buf[64];

    for (cc = ft_circuits; cc; cc = cc->ci_next)
        if (cc->ci_inprogress)
            ncc++;

    for (pl = plot_list; pl; pl = pl->pl_next)
        if (!pl->pl_written && pl->pl_dvecs)
            npl++;

    if (!ncc && !npl)
        return 1;

    fprintf(cp_out, "Warning: ");

    if (ncc) {
        fprintf(cp_out,
                "the following simulation%s still in progress:\n",
                (ncc > 1) ? "s are" : " is");
        for (cc = ft_circuits; cc; cc = cc->ci_next)
            if (cc->ci_inprogress)
                fprintf(cp_out, "\t%s\n", cc->ci_name);
    }

    if (ncc && npl)
        fprintf(cp_out, "and ");

    if (npl) {
        fprintf(cp_out,
                "the following plot%s been saved:\n",
                (npl > 1) ? "s haven't" : " hasn't");
        for (pl = plot_list; pl; pl = pl->pl_next)
            if (!pl->pl_written && pl->pl_dvecs)
                fprintf(cp_out, "%s\t%s, %s\n",
                        pl->pl_typename, pl->pl_title, pl->pl_name);
    }

    fprintf(cp_out, "\nAre you sure you want to quit (yes)? ");
    (void) fflush(cp_out);

    if (!fgets(buf, sizeof(buf), stdin)) {
        clearerr(stdin);
        *buf = 'y';
    }

    return ((*buf == 'y') || (*buf == 'Y') || (*buf == '\n'));
}
