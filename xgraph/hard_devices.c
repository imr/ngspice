/* $Header$ */
/*
 * Hardcopy Devices
 *
 * This file contains the basic output device table.  The hardcopy
 * dialog is automatically constructed from this table.
 *
 * $Log$
 * Revision 1.1  2004-01-25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.1.1.1  1999/12/03 23:15:52  heideman
 * xgraph-12.0
 *
 */
#ifndef lint
static char rcsid[] = "$Id$";

#endif

#include <stdio.h>
#include "copyright.h"
#include "xgout.h"
#include "hard_devices.h"
#include "params.h"

extern int hpglInit();
extern int psInit();
extern int idrawInit();
extern int tgifInit();

struct hard_dev hard_devices[] =
{
    {"HPGL", hpglInit, "lpr -P%s", "xgraph.hpgl", "paper",
     27.5, "1", 14.0, "1", 12.0, NONE},
    {"Postscript", psInit, "lpr -P%s", "xgraph.ps", "$PRINTER",
     19.0, "Times-Bold", 18.0, "Times-Roman", 12.0, NO},
    {"Idraw", idrawInit,
     "cat > /usr/tmp/idraw.tmp.ps; %s /usr/tmp/idraw.tmp.ps&",
     "~/.clipboard", "/usr/bin/X11/idraw", 19.0, "Times-Bold", 18.0,
     "Times-Roman", 12.0, NONE},
    {"Tgif", tgifInit,
     "cat > /usr/tmp/xgraph.obj; %s /usr/tmp/xgraph &",
     "xgraph.obj", "/usr/bin/X11/tgif", 19.0, "Times-Bold", 18.0,
     "Times-Roman", 12.0, NONE}
};

int     hard_count = sizeof(hard_devices) / sizeof(struct hard_dev);

#define CHANGE_D(name, field) \
if (param_get(name, &val)) { \
    if (val.type == DBL) { \
       hard_devices[idx].field = val.dblv.value; \
    } \
}

#define CHANGE_S(name, field) \
if (param_get(name, &val)) { \
    if (val.type == STR) { \
       (void) strcpy(hard_devices[idx].field, val.strv.value); \
    } \
}


void 
hard_init()
/*
 * Changes values in hard_devices structures in accordance with
 * parameters set using the parameters module.
 */
{
    char    nn[BUFSIZ];
    int     idx;
    params  val;

    for (idx = 0; idx < hard_count; idx++) {
	(void) sprintf(nn, "%s.Dimension", hard_devices[idx].dev_name);
	CHANGE_D(nn, dev_max_dim);
	(void) sprintf(nn, "%s.OutputTitleFont", hard_devices[idx].dev_name);
	CHANGE_S(nn, dev_title_font);
	(void) sprintf(nn, "%s.OutputTitleSize", hard_devices[idx].dev_name);
	CHANGE_D(nn, dev_title_size);
	(void) sprintf(nn, "%s.OutputAxisFont", hard_devices[idx].dev_name);
	CHANGE_S(nn, dev_axis_font);
	(void) sprintf(nn, "%s.OutputAxisSize", hard_devices[idx].dev_name);
	CHANGE_D(nn, dev_axis_size);
	if (hard_devices[idx].dev_printer[0] == '$') {
	    extern char *getenv();
	    char *ptr;
	    if ((ptr = getenv(&hard_devices[idx].dev_printer[1]))) {
		(void) strncpy(hard_devices[idx].dev_printer, ptr, MFNAME - 1);
		hard_devices[idx].dev_printer[MFNAME - 1] = '\0';
	    }
	}
    }
}
