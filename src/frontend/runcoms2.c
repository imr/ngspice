/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
$Id$
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
#include "runcoms2.h"
#include "runcoms.h"
#include "variable.h"
#include "breakp2.h"
#include "plotting/graf.h"

#include "inpdefs.h"

#define RAWBUF_SIZE 32768
extern char rawfileBuf[RAWBUF_SIZE];

/* Continue a simulation. If there is non in progress, this is the
 * equivalent of "run".
 */

/* ARGSUSED */

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
        com_run((wordlist *) NULL);
        return;
    }
    ft_curckt->ci_inprogress = TRUE;
    ft_setflag = TRUE;



    reset_trace( );
    for ( db = dbs, resumption = FALSE; db; db = db->db_next )
	if( db->db_type == DB_IPLOT || db->db_type == DB_IPLOTALL ) {
		resumption = TRUE;
	}

    /*rawfile output saj*/
    if (last_used_rawfile)
      dofile = TRUE;
    
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
    
    if (dofile) {
#ifdef PARALLEL_ARCH
      if (ARCHme == 0) {
#endif /* PARALLEL_ARCH */
        if (!last_used_rawfile)
	  rawfileFp = stdout;
        else if (!(rawfileFp = fopen(last_used_rawfile, "a"))) {
	  setvbuf(rawfileFp, rawfileBuf, _IOFBF, RAWBUF_SIZE);
	  perror(last_used_rawfile);
	  ft_setflag = FALSE;
	  return;
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
    



    /*end saj*/

    err = if_run(ft_curckt->ci_ckt, "resume", (wordlist *) NULL,
            ft_curckt->ci_symtab); 

    /*close rawfile saj*/
    if (rawfileFp){
      if (ftell(rawfileFp)==0) {
	(void) fclose(rawfileFp);
	(void) remove(last_used_rawfile);
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
    } else
        ft_curckt->ci_inprogress = FALSE;
    return;
}

/* Throw out the circuit struct and recreate it from the deck.  This command
 * should be obsolete.
 */

/* ARGSUSED */
void
com_rset(wordlist *wl)
{
    struct variable *v, *next;

    if (ft_curckt == NULL) {
        fprintf(cp_err, "Error: there is no circuit loaded.\n");
        return;
    }
    INPkillMods(); 

    if_cktfree(ft_curckt->ci_ckt, ft_curckt->ci_symtab);
    for (v = ft_curckt->ci_vars; v; v = next) {
	next = v->va_next;
	tfree(v);
    }
    ft_curckt->ci_vars = NULL;

    inp_dodeck(ft_curckt->ci_deck, ft_curckt->ci_name, (wordlist *) NULL,
            TRUE, ft_curckt->ci_options, ft_curckt->ci_filename);
    return;
}
