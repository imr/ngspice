/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "ftehelp.h"
#include "hlpdefs.h"
#include "misccoms.h"

#include "circuits.h"
#include "hcomp.h"
#include "variable.h"

#ifdef HAVE_GNUREADLINE
#include <readline/readline.h>
#include <readline/history.h>

extern int gnu_history_lines;
extern char gnu_history_file[];
#endif

static void byemesg(void);

void
com_quit(wordlist *wl)
{
    struct circ *cc;
    struct plot *pl;
    int ncc = 0, npl = 0;
    char buf[64];
    bool noask;

    (void) cp_getvar("noaskquit", VT_BOOL, (char *) &noask);
    gr_clean();
    cp_ccon(FALSE);
    
    /* Make sure the guy really wants to quit. */
    if (!ft_nutmeg && !noask) {
        for (cc = ft_circuits; cc; cc = cc->ci_next)
            if (cc->ci_inprogress)
                ncc++;
        for (pl = plot_list; pl; pl = pl->pl_next)
            if (!pl->pl_written && pl->pl_dvecs)
                npl++;
        if (ncc || npl) {
            fprintf(cp_out, "Warning: ");
            if (ncc) {
                fprintf(cp_out, 
            "the following simulation%s still in progress:\n",
                        (ncc > 1) ? "s are" : " is");
                for (cc = ft_circuits; cc; cc = cc->ci_next)
                    if (cc->ci_inprogress)
                        fprintf(cp_out, "\t%s\n",
                                cc->ci_name);
            }
            if (npl) {
                if (ncc)
                    fprintf(cp_out, "and ");
                fprintf(cp_out, 
                "the following plot%s been saved:\n",
                    (npl > 1) ? "s haven't" : " hasn't");
                for (pl = plot_list; pl; pl = pl->pl_next)
                    if (!pl->pl_written && pl->pl_dvecs)
                        fprintf(cp_out, "%s\t%s, %s\n",
                                pl->pl_typename,
                                pl->pl_title,
                                pl->pl_name);
            }
            fprintf(cp_out, 
                "\nAre you sure you want to quit (yes)? ");
            (void) fflush(cp_out);
            if (!fgets(buf, BSIZE_SP, stdin)) {
                clearerr(stdin);
                *buf = 'y';
            }
            if ((*buf == 'y') || (*buf == 'Y') || (*buf == '\n'))
                byemesg();
            else {
                return;
            }
        } else
            byemesg();
    } else
        byemesg();
#ifdef HAVE_GNUREADLINE
    /* Added GNU Readline Support -- Andrew Veliath <veliaa@rpi.edu> */
    if (cp_interactive && (cp_maxhistlength > 0)) {
	stifle_history(cp_maxhistlength);
	write_history(gnu_history_file);
    }
#endif /* HAVE_GNUREADLINE */
    exit(EXIT_NORMAL);
}


#ifdef SYSTEM_MAIL

void
com_bug(wordlist *wl)
{
    char buf[BSIZE_SP];

    if (!Bug_Addr || !*Bug_Addr) {
        fprintf(cp_err, "Error: No address to send bug reports to.\n");
	return;
    }
    fprintf(cp_out, "Calling the mail program . . .(sending to %s)\n\n",
	    Bug_Addr);
    fprintf(cp_out,
	    "Please include the OS version number and machine architecture.\n");
    fprintf(cp_out,
	    "If the problem is with a specific circuit, please include the\n");
    fprintf(cp_out, "input file.\n");

    (void) sprintf(buf, SYSTEM_MAIL, ft_sim->simulator,
	    ft_sim->version, Bug_Addr);
    (void) system(buf);
    fprintf(cp_out, "Bug report sent.  Thank you.\n");
    return;
}

#else


void
com_bug(wordlist *wl)
{
    fprintf(cp_out, "Send mail to the address ngspice-devel@lists.sourceforge.net\n");
    return;
}

#endif

void
com_version(wordlist *wl)
{
    char *s;

    if (!wl) {
	fprintf(cp_out, "******\n");

	fprintf(cp_out, "** %s-%s : %s\n", ft_sim->simulator,
		ft_sim->version, ft_sim->description);
	fprintf(cp_out, "** The U. C. Berkeley CAD Group\n");
	fprintf(cp_out,
	  "** Copyright 1985-1994, Regents of the University of California.\n");
	if (Spice_Notice && *Spice_Notice)
	    fprintf(cp_out, "** %s\n", Spice_Notice);
	if (Spice_Build_Date && *Spice_Build_Date)
	    fprintf(cp_out, "** Creation Date: %s\n", Spice_Build_Date);
	fprintf(cp_out, "******\n");

    } else {
        s = wl_flatten(wl);
	if (!strncmp(s, "-s", 2)) {
	    fprintf(cp_out, "******\n");
	    fprintf(cp_out, "** %s-%s\n", ft_sim->simulator,
		    ft_sim->version);
	    if (Spice_Notice && *Spice_Notice)
		fprintf(cp_out, "** %s\n", Spice_Notice);
	    if (Spice_Build_Date && *Spice_Build_Date)
		fprintf(cp_out, "** Creation Date: %s\n", Spice_Build_Date);
	    fprintf(cp_out, "******\n");
	} else if (!eq(ft_sim->version, s)) {
            fprintf(stderr,
        "Note: rawfile is version %s (current version is %s)\n",
                    wl->wl_word, ft_sim->version);
        }
        tfree(s);
    }
    return;
}

static void
byemesg(void)
{
    printf("%s-%s done\n", ft_sim->simulator, ft_sim->version);
    return;
}
