/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
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
#include "ngspice/trandefs.h"

#include "circuits.h"
#include "runcoms2.h"
#include "runcoms.h"
#include "variable.h"
#include "breakp2.h"
#include "plotting/graf.h"
#include "spiceif.h"
#include "outitf.h"
#include "numparam/numpaif.h"

#include "ngspice/inpdefs.h"

#if defined(XSPICE) && (defined(SIMULATOR) || defined(SHARED_MODULE))
#include "ngspice/evtproto.h"
#endif

extern void line_free_x(struct card *deck, bool recurse);
extern INPmodel *modtab;

#ifdef SHARED_MODULE
extern void exec_controls(wordlist *newcontrols);
#endif

#define line_free(line, flag)                   \
    do {                                        \
        line_free_x(line, flag);                \
        line = NULL;                            \
    } while(0)


/* Continue a simulation. If there is non in progress, this is the
 * equivalent of "run".
 */

/* This is a hack to tell iplot routine to redraw the grid and initialize
   the display device
*/

bool resumption = FALSE;


void
com_resume(wordlist *wl)
{
    struct dbcomm *db;
    int err;

    /*rawfile output saj*/
    bool dofile = FALSE;
    char buf[BSIZE_SP];
    bool ascii = AsciiRawFile;
    /*end saj*/

    NG_IGNORE(wl);

    /*saj fix segment*/
    if (!ft_curckt) {
        fprintf(cp_err, "Error: there aren't any circuits loaded.\n");
        return;
    } else if (ft_curckt->ci_ckt == NULL) { /* Set noparse? */
        fprintf(cp_err, "Error: circuit not parsed.\n");
        return;
    }
    /*saj*/

    if (ft_curckt->ci_inprogress == FALSE) {
        fprintf(cp_err, "Note: run starting\n");
        com_run(NULL);
        return;
    }
    ft_curckt->ci_inprogress = TRUE;
    ft_setflag = TRUE;

    reset_trace();
    for (db = dbs, resumption = FALSE; db; db = db->db_next)
        if (db->db_type == DB_IPLOT || db->db_type == DB_IPLOTALL)
            resumption = TRUE;

    /*rawfile output saj*/
    if (last_used_rawfile)
        dofile = TRUE;

    if (cp_getvar("filetype", CP_STRING, buf, sizeof(buf))) {
        if (eq(buf, "binary"))
            ascii = FALSE;
        else if (eq(buf, "ascii"))
            ascii = TRUE;
        else
            fprintf(cp_err,
                    "Warning: strange file type \"%s\" (using \"ascii\")\n", buf);
    }

    if (dofile) {
        if (!last_used_rawfile)
            rawfileFp = stdout;
#if defined(__MINGW32__) || defined(_MSC_VER)
        /* ask if binary or ASCII, open file with w or wb   hvogt 15.3.2000 */
        else if (ascii) {
            if ((rawfileFp = fopen(last_used_rawfile, "a")) == NULL) {
                perror(last_used_rawfile);
                ft_setflag = FALSE;
                return;
            }
        } else if (!ascii) {
            if ((rawfileFp = fopen(last_used_rawfile, "ab")) == NULL) {
                perror(last_used_rawfile);
                ft_setflag = FALSE;
                return;
            }
        }
        /*---------------------------------------------------------------------------*/
#else
        else if (!(rawfileFp = fopen(last_used_rawfile, "a"))) {
            perror(last_used_rawfile);
            ft_setflag = FALSE;
            return;
        }
#endif
        rawfileBinary = !ascii;
    } else {
        rawfileFp = NULL;
    } /* if dofile */

    /*end saj*/

    err = if_run(ft_curckt->ci_ckt, "resume", NULL,
                 ft_curckt->ci_symtab);

    /*close rawfile saj*/
    if (rawfileFp) {
        if (ftell(rawfileFp) == 0) {
            (void) fclose(rawfileFp);
            (void) unlink(last_used_rawfile);
        } else {
            (void) fclose(rawfileFp);
        }
    }
    /*end saj*/

    if (err == 1) {
        /* The circuit was interrupted somewhere. */

        fprintf(cp_err, "simulation interrupted\n");
    } else if (err == 2) {
        fprintf(cp_err, "simulation aborted\n");
        ft_curckt->ci_inprogress = FALSE;
    } else {
        ft_curckt->ci_inprogress = FALSE;
    }
}


/* Throw out the circuit struct and recreate it from the deck. */
void
com_rset(wordlist *wl)
{
    NG_IGNORE(wl);

    if (ft_curckt == NULL) {
        fprintf(cp_err, "Warning: there is no circuit loaded.\n");
        fprintf(cp_err, "    Command 'reset' is ignored.\n");
        return;
    }
    com_remcirc(NULL);
    inp_source_recent();
}


/* Clears ckt and removes current circuit from database */
void
com_remcirc(wordlist *wl)
{
    struct variable *v, *next;
    struct card *dd;     /*in: the spice deck */
    struct circ *p, *prev = NULL;

    NG_IGNORE(wl);

    if (ft_curckt == NULL) {
        fprintf(cp_err, "Warning: there is no circuit loaded.\n");
        fprintf(cp_err, "    Command 'remcirc' is ignored.\n");
        return;
    }

    /* delete numparam data structure dicoS */
    nupa_del_dicoS();
    /* delete entry in dicoslist */
    nupa_rem_dicoslist(ft_curckt->ci_dicos);

    dbfree(ft_curckt->ci_dbs);
    ft_curckt->ci_dbs = NULL;
    dbs = NULL;

    /* The next lines stem from com_rset */
    INPkillMods();

#if defined(XSPICE) && (defined(SIMULATOR) || defined(SHARED_MODULE))
    /* remove event queues, if XSPICE and not nutmeg */
    if (ft_curckt->ci_ckt)
        EVTunsetup(ft_curckt->ci_ckt);
#endif

    if_cktfree(ft_curckt->ci_ckt, ft_curckt->ci_symtab);
    for (v = ft_curckt->ci_vars; v; v = next) {
        next = v->va_next;
        tfree(v->va_name);
        if (v->va_type == CP_STRING)
            tfree(v->va_string);
        tfree(v);
    }
    ft_curckt->ci_vars = NULL;
    /* delete the deck, parameter list, and options list in ft_curckt */
    dd = ft_curckt->ci_deck;
    line_free(dd, TRUE);
    dd = ft_curckt->ci_param;
    line_free(dd, TRUE);
    dd = ft_curckt->ci_options;
    line_free(dd, TRUE);
    dd = ft_curckt->ci_meas;
    line_free(dd, TRUE);
    dd = ft_curckt->ci_auto;
    line_free(dd, TRUE);

    wl_free(ft_curckt->ci_commands);

    tfree(ft_curckt->FTEstats);

    ft_sim->deleteTask (ft_curckt->ci_ckt, ft_curckt->ci_defTask);
    if (ft_curckt->ci_specTask)
        ft_sim->deleteTask (ft_curckt->ci_ckt, ft_curckt->ci_specTask);

    if (ft_curckt->ci_name)
        tfree(ft_curckt->ci_name);
    if (ft_curckt->ci_filename)
        tfree(ft_curckt->ci_filename);
    rem_tlist(ft_curckt->devtlist);
    rem_tlist(ft_curckt->modtlist);

    inp_mc_free();

    /* delete the actual circuit entry from ft_circuits */
    for (p = ft_circuits; p; p = p->ci_next) {
        if (ft_curckt == p) {
            if (prev == NULL) {
                ft_circuits = p->ci_next;
                tfree(p);
                p = NULL;
                break;
            } else {
                prev->ci_next = p->ci_next;
                tfree(p);
                p = NULL;
                break;
            }
        }
        prev = p;
    }

    /* make first entry in ft_circuits the actual circuit (or NULL) */
    ft_curckt = ft_circuits;
    if (ft_curckt) {
        modtab = ft_curckt->ci_modtab;
        dbs = ft_curckt->ci_dbs;
        nupa_set_dicoslist(ft_curckt->ci_dicos);
    }
}
