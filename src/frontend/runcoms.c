/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Circuit simulation commands.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedebug.h"
#include "ngspice/dvec.h"

#include "numparam/numpaif.h"

#include "circuits.h"
#include "completion.h"
#include "runcoms.h"
#include "variable.h"
#include "spiceif.h"
#include "runcoms2.h"

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include ipc stuff */
#include "ngspice/ipctiein.h"
/* gtri - end - 12/12/90 */
#endif


static int dosim(char *what, wordlist *wl);
extern struct INPmodel *modtab;
extern struct dbcomm *dbs;
extern void NIresetwarnmsg(void);

/* Routines for the commands op, tran, ac, dc, listing, device, state,
 * resume, stop, trace, run, end.  Op, tran, ac, and dc cause the action
 * to be performed immediately, and run causes whatever actions were
 * present in the deck to be carried out. End has the effect of stopping
 * any simulations in progress, as opposed to ending the input deck as
 * the .end line does.
 */

FILE *rawfileFp;
bool rawfileBinary;
/*To tell resume the rawfile name saj*/
char *last_used_rawfile = NULL;
/*end saj */


/*
 * command "setcirc"
 *   print a list of circuits loaded
 * command "setcirc <n>"
 *   switch to circuit number <n>
 */

void
com_scirc(wordlist *wl)
{
    struct circ *p;
    int i, j = 0;

    if (ft_circuits == NULL) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return;
    }

    if (wl == NULL) {
        fprintf(cp_out,
                "List of circuits loaded:\n\n");
        for (p = ft_circuits; p; p = p->ci_next) {
            if (ft_curckt == p)
                fprintf(cp_out, "Current");
            fprintf(cp_out, "\t%d\t%s\n", ++j, p->ci_name);
        }
        return;
    } else {
        for (p = ft_circuits; p; p = p->ci_next)
            j++;

        p = NULL;
        if ((sscanf(wl->wl_word, " %d ", &i) != 1) || (i < 0) || (i > j))
            ;
        else
            for (p = ft_circuits; --i > 0; p = p->ci_next)
                ;
        /* for (p = ft_circuits; p; p = p->ci_next)
         *   if (ciprefix(wl->wl_word, p->ci_name))
         *    break;
         */
        if (p == NULL) {
            fprintf(cp_err, "Warning: no such circuit \"%s\"\n", wl->wl_word);
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
    /* get the model table for the current circuit, store it in the global variable modtab */
    modtab = ft_curckt->ci_modtab;
    /* get the database for save, iplot, stop */
    dbs = ft_curckt->ci_dbs;
    /* set the numparam dicos structure for use with measure */
    nupa_set_dicoslist(ft_curckt->ci_dicos);
}


void
com_pz(wordlist *wl)
{
    dosim("pz", wl);
}


void
com_op(wordlist *wl)
{
    dosim("op", wl);
}


void
com_dc(wordlist *wl)
{
    dosim("dc", wl);
}


void
com_ac(wordlist *wl)
{
    dosim("ac", wl);
}


void
com_tf(wordlist *wl)
{
    dosim("tf", wl);
}


void
com_tran(wordlist *wl)
{
    dosim("tran", wl);
}


void
com_sens(wordlist *wl)
{
    dosim("sens", wl);
}


void
com_disto(wordlist *wl)
{
    dosim("disto", wl);
}


void
com_noise(wordlist *wl)
{
    dosim("noise", wl);
}


#ifdef WITH_PSS
/* Steady State Analysis */
void
com_pss(wordlist *wl)
{
    dosim("pss", wl);
}
#endif

#ifdef RFSPICE
/* S-parameter Analysis*/
void
com_sp(wordlist* wl)
{
    dosim("sp", wl);
}
#endif

static int dosim(
        char *what, /* in: command
                     * (pz,op,dc,ac,tf,tran,sens,disto,noise,run) */
        wordlist *wl /* in: command option */
        /* global variables in: ft_curckt, ft_circuits,
         * out: ft_setflag, ft_intrpt, rawfileFp, rawfileBinary,
         *      last_used_rawfile
         */
    )
{
    wordlist *ww = NULL;
    bool dofile = FALSE;
    char buf[BSIZE_SP];
    struct circ *ct;
    int err = 0;
    /* set file type to binary or to what is given by environmental
       variable SPICE_ASCIIRAWFILE in ivars.c */
    bool ascii = AsciiRawFile;
    if (eq(what, "run") && wl) {
        dofile = TRUE;
    }
    /* add "what" to beginning of wordlist wl, except "what" equals "run"
       and a rawfile name is given (in wl) */
    if (!dofile) {
        ww = wl_cons(copy(what), wl);
    }
    /* reset output file type according to variable given in spinit */
    if (cp_getvar("filetype", CP_STRING, buf, sizeof(buf))) {
        if (eq(buf, "binary")) {
            ascii = FALSE;
        }
        else if (eq(buf, "ascii")) {
            ascii = TRUE;
        }
        else {
            fprintf(cp_err,
                    "Warning: strange file type \"%s\" (using \"ascii\")\n", buf);
            ascii = TRUE;
        }
    }

    if (!ft_curckt) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return 1;
    }
    else if (ft_curckt->ci_ckt == NULL) { /* Set noparse? */
        fprintf(cp_err, "Error: circuit not parsed.\n");
        return 1;
    }
    for (ct = ft_circuits; ct; ct = ct->ci_next) {
        if (ct->ci_inprogress && (ct != ft_curckt)) {
            fprintf(cp_err,
                    "Warning: losing old state for circuit '%s'\n",
                    ct->ci_name);
            ct->ci_inprogress = FALSE;
        }
    }
    /* "resume" will never occur in ngspice */
    if (ft_curckt->ci_inprogress && eq(what, "resume")) {
        ft_setflag = TRUE;  /* don't allow abort upon interrupt during run  */
        ft_intrpt = FALSE;
        fprintf(cp_err, "Warning: resuming run in progress.\n");
        com_resume(NULL);
        ft_setflag = FALSE;  /* Now allow aborts again  */
        return 0;
    }

    NIresetwarnmsg();

    /* From now on until the next prompt, an interrupt will just
     * set a flag and let spice finish up, then control will be
     * passed back to the user.
     */
    ft_setflag = TRUE;  /* Don't allow abort upon interrupt during run.  */
    ft_intrpt = FALSE;
    /* command "run" is given with rawfile name in wl */
    if (dofile) {
        if (!*wl->wl_word) {
            rawfileFp = stdout;
        }

        /* ask if binary or ASCII, open file with wb or w */
        else if (ascii) {
            if ((rawfileFp = fopen(wl->wl_word, "w")) == NULL) {
                perror(wl->wl_word);
                ft_setflag = FALSE;
                return 1;
            }
            fprintf(cp_out, "ASCII raw file \"%s\"\n", wl->wl_word);
        }
        else { /* binary */
            if ((rawfileFp = fopen(wl->wl_word, "wb")) == NULL) {
                perror(wl->wl_word);
                ft_setflag = FALSE;
                return 1;
            }
            fprintf(cp_out, "binary raw file \"%s\"\n", wl->wl_word);
        }
        rawfileBinary = !ascii;
    }
    else {
        rawfileFp = NULL;
    }

    /*save rawfile name */
    if (last_used_rawfile) {
        tfree(last_used_rawfile);
    }
    if (rawfileFp) {
        last_used_rawfile = copy(wl->wl_word);
    }
    else {
        last_used_rawfile = NULL;
    }

    ft_curckt->ci_inprogress = TRUE;
    cp_vset("sim_status", CP_NUM, &err);
    /* "sens2" not used in ngspice */
    if (eq(what, "sens2")) {
        if (if_sens_run(ft_curckt->ci_ckt, ww, ft_curckt->ci_symtab) == 1) {
            /* The circuit was interrupted somewhere. */
            fprintf(cp_err, "%s simulation interrupted\n", what);
#ifdef XSPICE
            /* gtri - add - 12/12/90 - wbk - record error and return errchk */
            g_ipc.run_error = IPC_TRUE;
            if (g_ipc.enabled) {
                ipc_send_errchk();
            }
            /* gtri - end - 12/12/90 */
#endif
        }
        else {
            ft_curckt->ci_inprogress = FALSE;
        }
        /* Do a run of the circuit */
    }
    else {
        err = if_run(ft_curckt->ci_ckt, what, ww, ft_curckt->ci_symtab);
        if (err == 1) {
            /* The circuit was interrupted somewhere. */
            fprintf(cp_err, "%s simulation interrupted\n", what);
#ifdef XSPICE
            /* record error and return errchk */
            g_ipc.run_error = IPC_TRUE;
            if (g_ipc.enabled) {
                ipc_send_errchk();
            }
            /* gtri - end - 12/12/90 */
#endif
            err = 0;
        }
        else if (err == 2) {
            fprintf(cp_err, "%s simulation(s) aborted\n", what);
            ft_curckt->ci_inprogress = FALSE;
            err = 1;
            cp_vset("sim_status", CP_NUM, &err);
        }
        else if (err == 3) {
            fprintf(cp_err, "%s simulation not started\n", what);
            ft_curckt->ci_inprogress = FALSE;
            err = 1;
            cp_vset("sim_status", CP_NUM, &err);
        }
        else {
            ft_curckt->ci_inprogress = FALSE;
        }
    }
    /* close the rawfile */
    if (rawfileFp) {
        if (ftell(rawfileFp) == 0) {
            (void) fclose(rawfileFp);
            if (wl) {
                (void) unlink(wl->wl_word);
            }
        }
        else {
            (void) fclose(rawfileFp);
        }
    }
    ft_curckt->ci_runonce = TRUE;
    ft_setflag = FALSE;

    /* va: garbage collection: unlink first word (inserted here) and tfree it */
    if (!dofile) {
        txfree(ww->wl_word);
        if (wl) {
            wl->wl_prev = NULL;
        }
        txfree(ww);
    }

    /* execute the .measure statements */
    if (!err && ft_curckt->ci_last_an && ft_curckt->ci_meas) {
        do_measure(ft_curckt->ci_last_an, FALSE);
    }

    return err;
} /* end of function dosim */



/* Usage is run [filename] */
void com_run(wordlist *wl)
{
    /* ft_getsaves(); */
    dosim("run", wl);
}


int
ft_dorun(char *file)
{
    static wordlist wl = { NULL, NULL, NULL };

    wl.wl_word = file;
    if (file)
        return dosim("run", &wl);
    else
        return dosim("run", NULL);
}


/* ARGSUSED */ /* until the else clause gets put back */
bool
ft_getOutReq(FILE **fpp, struct plot **plotp, bool *binp, char *name, char *title)
{
    NG_IGNORE(title);
    NG_IGNORE(name);
    NG_IGNORE(plotp);

    if (rawfileFp) {
        *fpp = rawfileFp;
        *binp = rawfileBinary;
        return (TRUE);
    } else {
        return (FALSE);
    }
}
