#include <config.h>
#include <stdio.h>

#include <ngspice.h>
#include <defines.h>
#include <bool.h>
#include <wordlist.h>
#include <cpdefs.h>
#include <fteinput.h>
#include <ftedev.h>

#include "plotting/plotit.h"
#include "plotting/graphdb.h"
#include "plotting/graf.h"

#include "arg.h"
#include "display.h"
#include "com_hardcopy.h"
#include "variable.h"


/* hardcopy file plotargs, or 'hardcopy file' -- with no other args
 * this prompts the user for a window to dump to a plot file. XXX no
 * it doesn't.  */
void
com_hardcopy(wordlist *wl)
{
    char *buf2;
    wordlist *process(wordlist *wlist);
    char *fname;
    char buf[BSIZE_SP], device[BSIZE_SP];
    bool tempf = FALSE;
    char *devtype;
    char format[513];
    int	printed;
    int hc_button;
    int foundit;

    if (!cp_getvar("hcopydev", VT_STRING, device))
        *device = '\0';

    if (wl) {
	hc_button = 0;
        fname = wl->wl_word;
        wl = wl->wl_next;
    } else {
	hc_button = 1;
        fname = smktemp("hc");
        tempf = TRUE;
    }

    if (!cp_getvar("hcopydevtype", VT_STRING, buf)) {
        devtype = "plot5";
    } else {
        devtype = buf;
    }

    /* enable screen plot selection for these display types */
    foundit = 0;

#ifndef X_DISPLAY_MISSING
    if (!wl && hc_button) {

        REQUEST request;
        RESPONSE response;
        GRAPH *tempgraph;
        
        request.option = click_option;
        Input(&request, &response);

        if (response.option == error_option) return;

	if (response.reply.graph) {

	    if (DevSwitch(devtype)) return;
	    tempgraph = CopyGraph(response.reply.graph);
	    tempgraph->devdep = fname;
	    if (NewViewport(tempgraph)) {
	      DevSwitch(NULL);
	      return;
	    }
	    gr_resize(tempgraph);
	    gr_redraw(tempgraph);
	    DestroyGraph(tempgraph->graphid);
	    DevSwitch(NULL);
	    foundit = 1;
	}
    }

#endif


    if (!foundit) {

	if (!wl) {
	    outmenuprompt("which variable ? ");
	    if ((buf2 = prompt(cp_in)) == (char *) -1)	/* XXXX Sick */
		return;
	    wl = (struct wordlist *) tmalloc(sizeof(struct wordlist));
	    wl->wl_word = buf2;
	    wl->wl_next = NULL;
	    wl = process(wl);
	}



	if (DevSwitch(devtype)) return;

	if (!wl || !plotit(wl, fname, (char *) NULL)) {
	    printf("com_hardcopy: graph not defined\n");
	    DevSwitch(NULL);    /* remember to switch back */
	    return;
	}

	DevSwitch(NULL);

    }

    printed = 0;


    if (*device) {
#ifdef SYSTEM_PLOT5LPR
      if (!strcmp(devtype, "plot5") || !strcmp(devtype, "MFB")) {
	if (!cp_getvar("lprplot5", VT_STRING, format))
		strcpy(format, SYSTEM_PLOT5LPR);
        (void) sprintf(buf, format, device, fname);
        fprintf(cp_out, "Printing %s on the %s printer.\n", fname, device);
        (void) system(buf);
	printed = 1;
      }
#endif
#ifdef SYSTEM_PSLPR
      if (!printed && !strcmp(devtype, "postscript")) {
        /* note: check if that was a postscript printer XXX */
	if (!cp_getvar("lprps", VT_STRING, format))
		strcpy(format, SYSTEM_PSLPR);
        (void) sprintf(buf, format, device, fname);
        fprintf(cp_out, "Printing %s on the %s printer.\n", fname, device);
        (void) system(buf);
	printed = 1;
      }
#endif
    }

    if (!printed) {
      if (!strcmp(devtype, "plot5")) {
        fprintf(cp_out,
	    "The file \"%s\" may be printed with the Unix \"plot\" command,\n",
                fname);
        fprintf(cp_out,
	    "\tor by using the '-g' flag to the Unix lpr command.\n");
      } else if (!strcmp(devtype, "postscript")) {
        fprintf(cp_out,
	    "The file \"%s\" may be printed on a postscript printer.\n",
	    fname);
      } else if (!strcmp(devtype, "MFB")) {
	fprintf(cp_out,
		"The file \"%s\" may be printed on a MFB device.\n",
		fname);
      }
    }

    if (tempf && *device)
        (void) unlink(fname);

    return;
}
