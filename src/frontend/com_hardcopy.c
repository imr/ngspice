#include "ngspice/config.h"
#include <stdio.h>

#include "ngspice/ngspice.h"
#include "ngspice/defines.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteinput.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedbgra.h"

#include "plotting/plotit.h"
#include "plotting/graphdb.h"
#include "plotting/graf.h"
#include "../misc/mktemp.h"

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
    char *fname;
    char buf[BSIZE_SP], device[BSIZE_SP];
    bool tempf = FALSE;
    char *devtype;
#if defined(SYSTEM_PLOT5LPR) || defined(SYSTEM_PSLPR)
    char format[513];
#endif
    int printed;
    int hc_button;
    int foundit;

    if (!cp_getvar("hcopydev", CP_STRING, device))
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

    if (!cp_getvar("hcopydevtype", CP_STRING, buf))
        devtype = "postscript";
    else
        devtype = buf;

    /* enable screen plot selection for these display types */
    foundit = 0;


    // PushGraphContext(currentgraph);

#ifdef HAS_WINDOWS
    if (!wl && hc_button) {
        char *psfname;
        GRAPH *tempgraph;
        if (DevSwitch(devtype))
            return;
        tempgraph = CopyGraph(currentgraph);
        /* change .tmp to .ps */
        psfname = strchr(fname, '.');
        if (psfname) {
            *(psfname + 1) = 'p';
            *(psfname + 2) = 's';
            *(psfname + 3) = '\0';
        } else {
            fname = realloc(fname, strlen(fname)+4);
            strcat(fname, ".ps");
        }
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
#endif


#ifndef X_DISPLAY_MISSING
    if (!wl && hc_button) {

        REQUEST request;
        RESPONSE response;
        GRAPH *tempgraph;

        request.option = click_option;
        Input(&request, &response);
        if (response.option == error_option)
            return;

        if (response.reply.graph) {
            if (DevSwitch(devtype))
                return;
            tempgraph = CopyGraph(response.reply.graph);
            tempgraph->devdep = fname;
            if (NewViewport(tempgraph)) {
                DevSwitch(NULL);
                return;
            }
            /* save current graphics context */
            PushGraphContext(currentgraph);
            currentgraph = tempgraph;
            /* some operations in gr_resize, gr_redraw, and DevSwitch
               will be done on currentgraph, not only on tempgraph */
            gr_resize(tempgraph);
            gr_redraw(tempgraph);
            DevSwitch(NULL);
            /* retrieve current graphics context */
            PopGraphContext();
            DestroyGraph(tempgraph->graphid);
            foundit = 1;
        }
    }

#endif

    /* save current graphics context, because plotit() will create a new
       currentgraph */
    PushGraphContext(currentgraph);

    if (!foundit) {

        if (!wl) {
            char *buf2;
            outmenuprompt("which variable ? ");
            buf2 = prompt(cp_in);
            if (!buf2)
                return;
            wl = wl_cons(buf2, NULL);
            wl = process(wl);
        }

        if (DevSwitch(devtype))
            return;

        if (!wl || !plotit(wl, fname, NULL)) {
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
            if (!cp_getvar("lprplot5", CP_STRING, format))
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
            if (!cp_getvar("lprps", CP_STRING, format))
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
                    "\nThe file \"%s\" may be printed on a postscript printer.\n",
                    fname);
        } else if (!strcmp(devtype, "MFB")) {
            fprintf(cp_out,
                    "The file \"%s\" may be printed on a MFB device.\n",
                    fname);
        }
    }

    if (tempf && *device)
        (void) unlink(fname);

    /* restore previous graphics context by retrieving the previous currentgraph */
    PopGraphContext();
}
