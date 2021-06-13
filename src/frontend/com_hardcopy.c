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
void com_hardcopy(wordlist *wl)
{
    char *fname = NULL;
    size_t n_byte_fname; /* size of fname in bytes, including null */
    char buf[BSIZE_SP], device[BSIZE_SP];
    bool tempf = FALSE;
    char *devtype;
#if defined(SYSTEM_PLOT5LPR) || defined(SYSTEM_PSLPR)
    char format[513];
#endif
    int printed;
    int hc_button;
    int foundit;

    static int n;

    if (!cp_getvar("hcopydev", CP_STRING, device, sizeof(device)))
        *device = '\0';

    if (!cp_getvar("hcopydevtype", CP_STRING, buf, sizeof(buf))) {
        devtype = "postscript";
    }
    else {
        devtype = buf;
    }

    if (wl) {
        hc_button = 0;
        fname = copy(wl->wl_word);
        n_byte_fname = (strlen(fname) + 1) * sizeof *fname;
        wl = wl->wl_next;
    }
    else {
        hc_button = 1;
        fname = smktemp2("hc", n);
        n++;
        tempf = TRUE;
        n_byte_fname = (strlen(fname) + 1) * sizeof *fname;
        if (!strcmp(devtype, "svg")) {
            fname = trealloc(fname, n_byte_fname + 4);
            (void)memcpy(fname + n_byte_fname - 1, ".svg", 5);
            n_byte_fname += 4;
        }
        else if (!strcmp(devtype, "postscript")) {
            fname = trealloc(fname, n_byte_fname + 3);
            (void)memcpy(fname + n_byte_fname - 1, ".ps", 4);
            n_byte_fname += 3;
        }
    }

    /* enable screen plot selection for these display types */
    foundit = 0;

#ifdef HAS_WINGUI
    if (!wl && hc_button) {
        char *psfname;
        GRAPH *tempgraph;
        /* initialze PS by calling PS_Init() */
        if (DevSwitch(devtype))
            return;
        if (currentgraph)
            tempgraph = CopyGraph(currentgraph);
        else {
            fprintf(stderr,
                    "No parameters for hardcopy command, not previous plot:\n");
            fprintf(stderr, "    Command hardcopy cannot be executed\n\n");
            DevSwitch(NULL);
            return;
        }

        if (!strcmp(devtype, "svg")) {
            /* change .tmp to .svg */
            psfname = strchr(fname, '.');
            if (psfname) {
                psfname[1] = 's';
                psfname[2] = 'v';
                psfname[3] = 'g';
                psfname[4] = '\0';
            }
            else {
                fname = trealloc(fname, n_byte_fname + 4);
                (void)memcpy(fname + n_byte_fname - 1, ".svg", 5);
                n_byte_fname += 4;
            }
        }
        else {
            /* change .tmp to .ps */
            psfname = strchr(fname, '.');
            if (psfname) {
                psfname[1] = 'p';
                psfname[2] = 's';
                psfname[3] = '\0';
            }
            else {
                fname = trealloc(fname, n_byte_fname + 3);
                (void)memcpy(fname + n_byte_fname - 1, ".ps", 4);
                n_byte_fname += 3;
            }
        }
        tempgraph->devdep = copy(fname);
        tempgraph->n_byte_devdep = n_byte_fname;

        if (NewViewport(tempgraph)) {
            DevSwitch(NULL);
            return;
        }
        gr_resize(tempgraph);
        /* use DevFinalize to add final statement in file, "/> or "stroke"*/
        DevFinalize();
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
        if (response.option == error_option) {
            return;
        }

        if (response.reply.graph) {
            if (DevSwitch(devtype)) {
                return;
            }
            tempgraph = CopyGraph(response.reply.graph);
            tempgraph->devdep = copy(fname);
            tempgraph->n_byte_devdep = n_byte_fname;
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
        if (!wl && cp_getvar("interactive", CP_BOOL, NULL, 0)) {
            char *buf2;
            outmenuprompt("which variable ? ");
            buf2 = prompt(cp_in);
            if (!buf2) {
                return;
            }
            wl = wl_cons(buf2, NULL);
            wl = process(wl);
        }

        if (DevSwitch(devtype)) {
            return;
        }

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
            if (!cp_getvar("lprplot5", CP_STRING, format, sizeof(format)))
                strcpy(format, SYSTEM_PLOT5LPR);
            (void) sprintf(buf, format, device, fname);
            if (system(buf) == -1) {
                fprintf(cp_out, "Printing %s on the %s printer failed.\n",
                        fname, device);
            }
            else {
                fprintf(cp_out, "Printing %s on the %s printer OK.\n",
                        fname, device);
                printed = 1;
            }
        }
#endif
#ifdef SYSTEM_PSLPR
        if (!printed && !strcmp(devtype, "postscript")) {
            /* note: check if that was a postscript printer XXX */
            if (!cp_getvar("lprps", CP_STRING, format, sizeof(format)))
                strcpy(format, SYSTEM_PSLPR);
            (void) sprintf(buf, format, device, fname);
            if (system(buf) == -1) {
                fprintf(cp_out, "Printing %s on the %s printer failed.\n",
                        fname, device);
            }
            else {
                fprintf(cp_out, "Printing %s on the %s printer OK.\n",
                        fname, device);
                printed = 1;
            }
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
        }
        else if (!strcmp(devtype, "postscript")) {
            fprintf(cp_out,
                    "\nThe file \"%s\" may be printed on a postscript printer.\n",
                    fname);
        }
        else if (!strcmp(devtype, "svg")) {
            fprintf(cp_out,
                "\nThe file \"%s\" has the Scalable Vector Graphics format.\n",
                fname);
        }
        else if (!strcmp(devtype, "MFB")) {
            fprintf(cp_out,
                    "The file \"%s\" may be printed on a MFB device.\n",
                    fname);
        }
    }

    if (tempf && *device) {
        (void) unlink(fname);
    }

    tfree(fname);

    /* restore previous graphics context by retrieving the previous currentgraph */
    PopGraphContext();
} /* end of function com_hardcopy */



