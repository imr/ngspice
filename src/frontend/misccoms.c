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


static void byemesg(void);

static int
hcomp(const void *a, const void *b)
{
    struct comm **c1 = (struct comm **) a;
    struct comm **c2 = (struct comm **) b;
    return (strcmp((*c1)->co_comname, (*c2)->co_comname));
}



void
com_help(wordlist *wl)
{
    struct comm *c;
    struct comm *ccc[512];  /* Should be enough. */
    int numcoms, i;
    bool allflag = FALSE;

    if (wl && eq(wl->wl_word, "all")) {
        allflag = TRUE;
        wl = NULL;  /* XXX Probably right */
    }

    /* We want to use more mode whether "moremode" is set or not. */
    out_moremode = TRUE;
    out_init();
    out_moremode = FALSE;
    if (wl == NULL) {
	out_printf(
	"For a complete description read the Spice3 User's Manual manual.\n");

	if (!allflag) {
	    out_printf(
		"For a list of all commands type \"help all\", for a short\n");
	    out_printf(
		"description of \"command\", type \"help command\".\n");
	}

        /* Sort the commands */
        for (numcoms = 0; cp_coms[numcoms].co_func != NULL; numcoms++)
            ccc[numcoms] = &cp_coms[numcoms];
        qsort((char *) ccc, numcoms, sizeof (struct comm *), hcomp);

        for (i = 0; i < numcoms; i++) {
            if ((ccc[i]->co_spiceonly && ft_nutmeg) || 
                    (ccc[i]->co_help == NULL) || 
                    (!allflag && !ccc[i]->co_major))
                continue;
            out_printf("%s ", ccc[i]->co_comname);
            out_printf(ccc[i]->co_help, cp_program);
            out_send("\n");
        }
    } else {
        while (wl != NULL) {
            for (c = &cp_coms[0]; c->co_func != NULL; c++)
                if (eq(wl->wl_word, c->co_comname)) {
                    out_printf("%s ", c->co_comname);
                    out_printf(c->co_help, cp_program);
                    if (c->co_spiceonly && ft_nutmeg)
                        out_send(
                        " (Not available in nutmeg)");
                    out_send("\n");
                    break;
                }
            if (c->co_func == NULL) {
                /* See if this is aliased. */
                struct alias *al;

                for (al = cp_aliases; al; al = al->al_next)
                    if (eq(al->al_name, wl->wl_word))
                        break;
                if (al == NULL)
                    fprintf(cp_out, 
                        "Sorry, no help for %s.\n", 
                        wl->wl_word);
                else {
                    out_printf("%s is aliased to ",
                        wl->wl_word);
                    /* Minor badness here... */
                    wl_print(al->al_text, cp_out);
                    out_send("\n");
                }
            }
            wl = wl->wl_next;
        }
    }
    out_send("\n");
    return;
}

void
com_ahelp(wordlist *wl)
{

    int i, n;
    /* assert: number of commands must be less than 512 */
    struct comm *cc[512];
    int env = 0;
    struct comm *com;
    int level;
    char slevel[256];

    if (wl) {
      com_help(wl);
      return;
    }

    out_init();

    /* determine environment */
    if (plot_list->pl_next) {   /* plots load */
      env |= E_HASPLOTS;
    } else {
      env |= E_NOPLOTS;
    }

    /* determine level */
    if (cp_getvar("level", VT_STRING, slevel)) {
      switch (*slevel) {
        case 'b':   level = 1;
            break;
        case 'i':   level = 2;
            break;
        case 'a':   level = 4;
            break;
        default:    level = 1;
            break;
      }
    } else {
      level = 1;
    }

    out_printf(
	"For a complete description read the Spice3 User's Manual manual.\n");
    out_printf(
	"For a list of all commands type \"help all\", for a short\n");
    out_printf(
	"description of \"command\", type \"help command\".\n");

    /* sort the commands */
    for (n = 0; cp_coms[n].co_func != (void (*)()) NULL; n++) {
      cc[n] = &cp_coms[n];
    }
    qsort((char *) cc, n, sizeof(struct comm *), hcomp);

    /* filter the commands */
    for (i=0; i< n; i++) {
      com = cc[i];
      if ((com->co_env < (level << 13)) && (!(com->co_env & 4095) ||
        (env & com->co_env))) {
        if ((com->co_spiceonly && ft_nutmeg) ||
        (com->co_help == (char *) NULL)) {
          continue;
        }
        out_printf("%s ", com->co_comname);
        out_printf(com->co_help, cp_program);
        out_send("\n");
      }
    }

    out_send("\n");

    return;

}

void
com_ghelp(wordlist *wl)
{
    char *npath, *path = Help_Path, buf[BSIZE_SP];
    int i;

    if (cp_getvar("helppath", VT_STRING, buf))
        path = copy(buf);
    if (!path) {
        fprintf(cp_err, "Note: defaulting to old help.\n\n");
        com_help(wl);
        return;
    }
    if (!(npath = cp_tildexpand(path))) {
        fprintf(cp_err, "Note: can't find help dir %s\n", path);
        fprintf(cp_err, "Defaulting to old help.\n\n");
        com_help(wl);
        return;
    }
    path = npath;
    if (cp_getvar("helpregfont", VT_STRING, buf))
        hlp_regfontname = copy(buf);
    if (cp_getvar("helpboldfont", VT_STRING, buf))
        hlp_boldfontname = copy(buf);
    if (cp_getvar("helpitalicfont", VT_STRING, buf))
        hlp_italicfontname = copy(buf);
    if (cp_getvar("helptitlefont", VT_STRING, buf))
        hlp_titlefontname = copy(buf);
    if (cp_getvar("helpbuttonfont", VT_STRING, buf))
        hlp_buttonfontname = copy(buf);
    if (cp_getvar("helpinitxpos", VT_NUM, (char *) &i))
        hlp_initxpos = i;
    if (cp_getvar("helpinitypos", VT_NUM, (char *) &i))
        hlp_initypos = i;
    if (cp_getvar("helpbuttonstyle", VT_STRING, buf)) {
        if (cieq(buf, "left"))
            hlp_buttonstyle = BS_LEFT;
        else if (cieq(buf, "center"))
            hlp_buttonstyle = BS_CENTER;
        else if (cieq(buf, "unif"))
            hlp_buttonstyle = BS_UNIF;
        else
            fprintf(cp_err, "Warning: no such button style %s\n",
                    buf);
    }
    if (cp_getvar("width", VT_NUM, (char *) &i))
        hlp_width = i;
    if (cp_getvar("display", VT_STRING, buf))
        hlp_displayname = copy(buf);
    else if (cp_getvar("device", VT_STRING, buf))
        hlp_displayname = copy(buf);
    else
        hlp_displayname = NULL;
    hlp_main(path, wl);
    return;
}


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

    exit(EXIT_NORMAL);

}


#ifdef SYSTEM_MAIL
#define MAIL_BUGS_
#endif

#ifdef MAIL_BUGS_


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
    fprintf(cp_out, "Send mail to the address ng-spice@ieee.ing.uniroma1.it\n");
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
