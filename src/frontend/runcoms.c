/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Circuit simulation commands.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "ftedev.h"
#include "ftedebug.h"
#include "dvec.h"

#include "circuits.h"
#include "completion.h"
#include "runcoms.h"
#include "variable.h"

/* static declarations */
static int dosim(char *what, wordlist *wl);

extern struct dbcomm *dbs;

/* Routines for the commands op, tran, ac, dc, listing, device, state,
 * resume, stop, trace, run, end.  Op, tran, ac, and dc cause the action
 * to be performed immediately, and run causes whatever actions were
 * present in the deck to be carried out. End has the effect of stopping
 * any simulations in progress, as opposed to ending te input deck as
 * the .end line does.
 */

FILE *rawfileFp;
bool rawfileBinary;
#define RAWBUF_SIZE 32768
char rawfileBuf[RAWBUF_SIZE];

void
com_scirc(wordlist *wl)
{
    struct circ *p;
    int i, j = 0;
    char buf[BSIZE_SP];

    if (ft_circuits == NULL) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return;
    }
    if (wl == NULL) {
        fprintf(cp_out, 
            "\tType the number of the desired circuit:\n\n");
        for (p = ft_circuits; p; p = p->ci_next) {
            if (ft_curckt == p)
                fprintf(cp_out, "Current");
            fprintf(cp_out, "\t%d\t%s\n", ++j, p->ci_name);
        }
        fprintf(cp_out, "? ");
        (void) fflush(cp_out);
        (void) fgets(buf, BSIZE_SP, cp_in);
        clearerr(cp_in);
        if ((sscanf(buf, " %d ", &i) != 1) || (i < 0) || (i > j))
            return;
        for (p = ft_circuits; --i > 0; p = p->ci_next);
    } else {
        for (p = ft_circuits; p; p = p->ci_next)
            if (ciprefix(wl->wl_word, p->ci_name))
                break;
        if (p == NULL) {
            fprintf(cp_err, "Warning: no such circuit \"%s\"\n",
                    wl->wl_word);
            return;
        }
        fprintf(cp_out, "\t%s\n", p->ci_name);
    }
    if (ft_curckt) {
        /* Actually this can't be FALSE */
        ft_curckt->ci_devices = 
                cp_kwswitch(CT_DEVNAMES, p->ci_devices);
        ft_curckt->ci_nodes = cp_kwswitch(CT_NODENAMES, p->ci_nodes);
    }
    ft_curckt = p;
    return;
}

void
com_pz(wordlist *wl)
{
    dosim("pz", wl);
    return;
}

void
com_op(wordlist *wl)
{
    dosim("op", wl);
    return;
}

void
com_dc(wordlist *wl)
{
    dosim("dc", wl);
    return;
}

void
com_ac(wordlist *wl)
{
    dosim("ac", wl);
    return;
}

void
com_tf(wordlist *wl)
{
    dosim("tf", wl);
    return;
}

void
com_tran(wordlist *wl)
{
    dosim("tran", wl);
    return;
}

void
com_sens(wordlist *wl)
{
    dosim("sens", wl);
    return;
}

void
com_disto(wordlist *wl)
{
    dosim("disto", wl);
    return;
}

void
com_noise(wordlist *wl)
{
    dosim("noise", wl);
    return;
}

static int
dosim(char *what, wordlist *wl)
{
    wordlist *ww = NULL;
    bool dofile = FALSE;
    char buf[BSIZE_SP];
    struct circ *ct;
    int err = 0;
    bool ascii = AsciiRawFile;

    if (eq(what, "run") && wl)
        dofile = TRUE;
    if (!dofile) {
        ww = alloc(struct wordlist);
        ww->wl_next = wl;
        if (wl)
            wl->wl_prev = ww;
        ww->wl_word = copy(what);
    }

    if (cp_getvar("filetype", VT_STRING, buf)) {
        if (eq(buf, "binary"))
            ascii = FALSE;
        else if (eq(buf, "ascii"))
	    ascii = TRUE;
	else
            fprintf(cp_err,
		    "Warning: strange file type \"%s\" (using \"ascii\")\n",
                    buf);
    }

    if (!ft_curckt) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return 1;
    } else if (ft_curckt->ci_ckt == NULL) { /* Set noparse? */
        fprintf(cp_err, "Error: circuit not parsed.\n");
        return 1;
    }
    for (ct = ft_circuits; ct; ct = ct->ci_next)
        if (ct->ci_inprogress && (ct != ft_curckt)) {
            fprintf(cp_err, 
                "Warning: losing old state for circuit '%s'\n",
                ct->ci_name);
            ct->ci_inprogress = FALSE;
        }
    if (ft_curckt->ci_inprogress && eq(what, "resume")) {
        ft_setflag = TRUE;
        ft_intrpt = FALSE;
        fprintf(cp_err, "Warning: resuming run in progress.\n");
        com_resume((wordlist *) NULL);
        ft_setflag = FALSE;
        return 0;
    }

    /* From now on until the next prompt, an interrupt will just
     * set a flag and let spice finish up, then control will be
     * passed back to the user.
     */
    ft_setflag = TRUE;
    ft_intrpt = FALSE;
    if (dofile) {
#ifdef PARALLEL_ARCH
      if (ARCHme == 0) {
#endif /* PARALLEL_ARCH */
        if (!*wl->wl_word)
	    rawfileFp = stdout;
        else if (!(rawfileFp = fopen(wl->wl_word, "w"))) {
        	setvbuf(rawfileFp, rawfileBuf, _IOFBF, RAWBUF_SIZE);
            perror(wl->wl_word);
            ft_setflag = FALSE;
            return 1;
        }
        rawfileBinary = !ascii;
#ifdef PARALLEL_ARCH
      } else {
        rawfileFp = NULL;
      }
#endif /* PARALLEL_ARCH */
    } else {
        rawfileFp = NULL;
    }

    /* Spice calls wrd_init and wrd_end itself */
    ft_curckt->ci_inprogress = TRUE;
    if (eq(what,"sens2")) {
       if (if_sens_run(ft_curckt->ci_ckt, ww, ft_curckt->ci_symtab) == 1) {
        /* The circuit was interrupted somewhere. */

	    fprintf(cp_err, "%s simulation interrupted\n", what);
        } else
	    ft_curckt->ci_inprogress = FALSE;
    } else {
        err = if_run(ft_curckt->ci_ckt, what, ww, ft_curckt->ci_symtab);
        if (err == 1) {
	    /* The circuit was interrupted somewhere. */
	    fprintf(cp_err, "%s simulation interrupted\n", what);
	    err = 0;
        } else if (err == 2) {
	    fprintf(cp_err, "%s simulation(s) aborted\n", what);
	    ft_curckt->ci_inprogress = FALSE;
	    err = 1;
	} else
	    ft_curckt->ci_inprogress = FALSE;
    }
    if (rawfileFp){
      if (ftell(rawfileFp)==0) {
	    (void) fclose(rawfileFp);
            (void) remove(wl->wl_word);
        } else {
	    (void) fclose(rawfileFp);
        }
    }
    ft_curckt->ci_runonce = TRUE;
    ft_setflag = FALSE;
    return err;
}

/* Usage is run [filename] Do the wrd_{open,run} and wrd_(void) close
 * here, and let the CKT stuff do wrd_init and wrd_end.
 */

void
com_run(wordlist *wl)
{
/*    ft_getsaves(); */
    dosim("run", wl);
    return;
}

int
ft_dorun(char *file)
{
    static wordlist wl = { NULL, NULL, NULL } ;

    wl.wl_word = file;
    if (file)
        return dosim("run", &wl);
    else
	return dosim("run", (wordlist *) NULL);
}

/* ARGSUSED */ /* until the else clause gets put back */
bool
ft_getOutReq(FILE **fpp, struct plot **plotp, bool *binp, char *name, char *title)
{
    /*struct plot *pl;*/
#ifndef BATCH

    if ( (strcmp(name, "Operating Point")==0) ||
         (strcmp(name, "AC Operating Point")==0) ) {
        return (FALSE);
    };
#endif

    if (rawfileFp) {
        *fpp = rawfileFp;
        *binp = rawfileBinary;
        return (TRUE);
    } else {
/*
        pl = plot_alloc(name);
        pl->pl_title = copy(title);
        pl->pl_name = copy(name);
        pl->pl_date = copy(datestring( ));

        pl->pl_next = plot_list;
        plot_list = pl;
        plot_cur = pl;

        *plotp = pl;
*/
        return (FALSE);
    }
}

