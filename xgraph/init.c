/* $Header$ */
/*
 * init.c: xgraph initialization code
 *
 * Routines:
 *	InitSets();
 *	ParseArgs();
 *	ReadDefaults();
 *
 * $Log$
 * Revision 1.1  2004-01-25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.2  1999/12/19 00:52:06  heideman
 * warning suppresion, slightly different flot ahndling
 *
 * Revision 1.1.1.1  1999/12/03 23:15:53  heideman
 * xgraph-12.0
 *
 */
#ifndef lint
static char rcsid[] = "$Id$";
#endif

#include "copyright.h"
#include <stdio.h>
#include <math.h>
#include <pwd.h>
#include <ctype.h>
#include "xgraph.h"
#include "xtb.h"
#include "hard_devices.h"
#include "params.h"

/*
 * Default settings for xgraph parameters
 */

/* PW */
#define DEF_ANIMATE             "off"
#define DEF_DELAY_VALUE         "2"
#define DEF_BORDER_WIDTH	"2"
#define DEF_BORDER_COLOR	"Black"
#define DEF_TITLE_TEXT		"X Graph"
#define DEF_XUNIT_TEXT		"X"
#define DEF_YUNIT_TEXT		"Y"
#define DEF_TICK_FLAG		"off"
#define DEF_TICKAXIS_FLAG	"off"
#define DEF_MARK_FLAG		"off"
#define DEF_PIXMARK_FLAG	"off"
#define DEF_LARGEPIX_FLAG	"off"
#define DEF_DIFFMARK_FLAG	"off"
#define DEF_BB_FLAG		"off"
#define DEF_NOLINE_FLAG		"off"
#define DEF_NOLEGEND_FLAG	"off"
#define DEF_NOBUTTON_FLAG	"off"
#define DEF_LOGX_FLAG		"off"
#define DEF_LOGY_FLAG		"off"
#define DEF_BAR_FLAG		"off"
#define DEF_STK_FLAG		"off"
#define DEF_FITX_FLAG		"off"
#define DEF_FITY_FLAG		"off"
#define DEF_BAR_BASE		"0.0"
#define DEF_BAR_OFFS		"0.0"
#define DEF_BAR_WIDTH		"-1.0"
#define DEF_LINE_WIDTH		"0"
#define DEF_GRID_SIZE		"0"
#define DEF_GRID_STYLE		"10"
#define DEF_LABEL_FONT		"helvetica-10"
#define DEF_TITLE_FONT		"helvetica-18"
#define DEF_GEOMETRY		""
#define DEF_REVERSE		"off"
/* PW Changes these to please JM */
#define DEF_FMT_X		"%.4f"
#define DEF_FMT_Y		"%.4f"
#define DEF_DEVICE		""
#define DEF_OUTPUT_DEVICE	D_XWINDOWS
#define DEF_DISPOSITION		"To Device"
#define DEF_FILEORDEV		""
#define DEF_DOCUMENT		"off"
#define DEF_SCALE		"1.0"

#define DEF_MARKER_FLAG		"off"
#define DEF_DIFFMARK_FLAG	"off"
#define DEF_PIXMARK_FLAG	"off"
#define DEF_LARGEPIX_FLAG	"off"

/* Low > High means set it based on the data */
#define DEF_LOW_LIMIT		"1.0"
#define DEF_HIGH_LIMIT		"0.0"

/* Black and white defaults */
#define DEF_BW_BACKGROUND	"white"
#define DEF_BW_BORDER		"black"
#define DEF_BW_ZEROCOLOR	"black"
#define DEF_BW_ZEROWIDTH	"3"
#define DEF_BW_ZEROSTYLE	"1"
#define DEF_BW_FOREGROUND	"black"

/* Color defaults */
#define DEF_COL_BACKGROUND	"#ccc"
#define DEF_COL_BORDER		"black"
#define DEF_COL_ZEROCOLOR	"white"
#define DEF_COL_ZEROWIDTH	"0"
#define DEF_COL_ZEROSTYLE	"1"
#define DEF_COL_FOREGROUND	"black"
#define DEF_COL_FIRSTSTYLE	"1"

/* Default line styles */
static char *defStyle[MAXATTR] =
{
    "1", "10", "11110000", "010111", "1110",
    "1111111100000000", "11001111", "0011000111"
};

/* Default color names */
/*static char *defColors[MAXATTR] =
{
    "red", "SpringGreen", "blue", "yellow",
    "cyan", "sienna", "orange", "coral"
};*/
static char *defColors[MAXATTR] =
{
    "red", "SpringGreen", "blue", "yellow",
    "purple", "orange", "hotpink", "cyan"
};

void
InitSets(o)
int     o;

/*
 * Initializes the data sets with default information.  Sets up
 * original values for parameters in parameters package.
 */
{
    int     idx;
    char    buf[1024];

    if (o == D_XWINDOWS) {
	/*
	 * Used to do all kinds of searching through visuals, etc. Got
	 * complaints -- so back to the simple version.
	 */
	vis = DefaultVisual(disp, DefaultScreen(disp));
	cmap = DefaultColormap(disp, DefaultScreen(disp));
	screen = DefaultScreen(disp);
	depth = DefaultDepth(disp, DefaultScreen(disp));
	param_init(disp, cmap);
    }
    else
	param_init(NULL, 0);


    param_set("Debug", BOOL, "false");
    param_set("Geometry", STR, DEF_GEOMETRY);
    param_set("ReverseVideo", BOOL, DEF_REVERSE);

    param_set("BorderSize", INT, DEF_BORDER_WIDTH);
    param_set("TitleText", STR, DEF_TITLE_TEXT);
    param_set("XUnitText", STR, DEF_XUNIT_TEXT);
    param_set("YUnitText", STR, DEF_YUNIT_TEXT);	/* YUnits */
    param_set("Ticks", BOOL, DEF_TICK_FLAG);
    param_set("TickAxis", BOOL, DEF_TICKAXIS_FLAG);

    param_set("Markers", BOOL, DEF_MARKER_FLAG);	/* markFlag (-m) */
    param_set("StyleMarkers", BOOL, DEF_DIFFMARK_FLAG);	/* colorMark (-M) */
    param_set("PixelMarkers", BOOL, DEF_PIXMARK_FLAG);	/* pixelMarks  (-p) */
    param_set("LargePixels", BOOL, DEF_LARGEPIX_FLAG);	/* bigPixel (-P) */

    param_set("BoundBox", BOOL, DEF_BB_FLAG);
    param_set("NoLines", BOOL, DEF_NOLINE_FLAG);
    param_set("NoLegend", BOOL, DEF_NOLEGEND_FLAG);
    param_set("NoButton", BOOL, DEF_NOBUTTON_FLAG);
    param_set("LogX", BOOL, DEF_LOGX_FLAG);
    param_set("LogY", BOOL, DEF_LOGY_FLAG);	/* logYFlag */
    param_set("BarGraph", BOOL, DEF_BAR_FLAG);
    param_set("StackGraph", BOOL, DEF_STK_FLAG);
    param_set("FitX", BOOL, DEF_FITX_FLAG);
    param_set("FitY", BOOL, DEF_FITY_FLAG);
    param_set("BarBase", DBL, DEF_BAR_BASE);
    param_set("BarWidth", DBL, DEF_BAR_WIDTH);
    param_set("BarOffset", DBL, DEF_BAR_OFFS);
    param_set("LineWidth", INT, DEF_LINE_WIDTH);
    param_set("GridSize", INT, DEF_GRID_SIZE);
    param_set("GridStyle", STYLE, DEF_GRID_STYLE);
    param_set("Format X", STR, DEF_FMT_X);
    param_set("Format Y", STR, DEF_FMT_Y);

    param_set("Device", STR, DEF_DEVICE);
    param_set("Disposition", STR, DEF_DISPOSITION);
    param_set("FileOrDev", STR, DEF_FILEORDEV);
    sprintf(buf, "%d", o);
    param_set("Output Device", INT, buf);
    param_set("Document", BOOL, DEF_DOCUMENT);
    param_set("Scale", DBL, DEF_SCALE);

    /* Set the user bounding box */
    param_set("XLowLimit", DBL, DEF_LOW_LIMIT);
    param_set("YLowLimit", DBL, DEF_LOW_LIMIT);
    param_set("XHighLimit", DBL, DEF_HIGH_LIMIT);
    param_set("YHighLimit", DBL, DEF_HIGH_LIMIT);

    /* Depends critically on whether the display has color */
    if (depth < 4) {
	/* Its black and white */
	param_set("Background", PIXEL, DEF_BW_BACKGROUND);
	param_set("Border", PIXEL, DEF_BW_BORDER);
	param_set("ZeroColor", PIXEL, DEF_BW_ZEROCOLOR);
	param_set("ZeroWidth", INT, DEF_BW_ZEROWIDTH);
	param_set("ZeroStyle", STYLE, DEF_BW_ZEROSTYLE);
	param_set("Foreground", PIXEL, DEF_BW_FOREGROUND);
	/* Initialize set defaults */
	for (idx = 0; idx < MAXATTR; idx++) {
	    (void) sprintf(buf, "%d.Style", idx);
	    param_set(buf, STYLE, defStyle[idx]);
	    (void) sprintf(buf, "%d.Color", idx);
	    param_set(buf, PIXEL, DEF_BW_FOREGROUND);
	}
    }
    else {
	/* Its color */
	param_set("Background", PIXEL, DEF_COL_BACKGROUND);
	param_set("Border", PIXEL, DEF_COL_BORDER);
	param_set("ZeroColor", PIXEL, DEF_COL_ZEROCOLOR);
	param_set("ZeroWidth", INT, DEF_COL_ZEROWIDTH);
	param_set("ZeroStyle", STYLE, DEF_COL_ZEROSTYLE);
	param_set("Foreground", PIXEL, DEF_COL_FOREGROUND);
	/* Initalize attribute colors defaults */
	for (idx = 0; idx < MAXATTR; idx++) {
	    (void) sprintf(buf, "%d.Style", idx);
	    param_set(buf, STYLE, defStyle[idx]);
	    (void) sprintf(buf, "%d.Color", idx);
	    param_set(buf, PIXEL, defColors[idx]);
	}
    }

    param_set("LabelFont", FONT, DEF_LABEL_FONT);
    param_set("TitleFont", FONT, DEF_TITLE_FONT);
    /* PW */
    param_set("Animate", BOOL, DEF_ANIMATE);
    param_set("DelayValue", INT, DEF_DELAY_VALUE);

    /* Initialize the data sets */
    for (idx = 0; idx < MAXSETS; idx++) {
	(void) sprintf(buf, "Set %d", idx);
	PlotData[idx].setName = STRDUP(buf);
	PlotData[idx].list = (PointList *) 0;
    }
}



static char *def_str;

#define DEF(name, type) \
if (def_str = XGetDefault(disp, Prog_Name, name)) { \
    param_set(name, type, def_str); \
}

void
ReadDefaults()
/*
 * Reads X default values which override the hard-coded defaults
 * set up by InitSets.
 */
{
    char    newname[100];
    int     idx;

    DEF("Debug", BOOL);
    DEF("Geometry", STR);
    DEF("Background", PIXEL);
    DEF("BorderSize", INT);
    DEF("Border", PIXEL);
    DEF("GridSize", INT);
    DEF("GridStyle", STYLE);
    DEF("Foreground", PIXEL);
    DEF("ZeroColor", PIXEL);
    DEF("ZeroStyle", STYLE);
    DEF("ZeroWidth", INT);
    DEF("LabelFont", FONT);
    DEF("TitleFont", FONT);
    DEF("Ticks", BOOL);
    DEF("TickAxis", BOOL);
    DEF("Device", STR);
    DEF("Disposition", STR);
    DEF("FileOrDev", STR);
    DEF("PixelMarkers", BOOL);
    DEF("LargePixels", BOOL);
    DEF("Markers", BOOL);
    DEF("StyleMarkers", BOOL);
    DEF("BoundBox", BOOL);
    DEF("NoLines", BOOL);
    DEF("LineWidth", INT);
    /* PW */
    DEF("Animate",BOOL);
    DEF("DelayValue",INT);
    /* End PW */

    /* Read device specific parameters */
    for (idx = 0; idx < hard_count; idx++) {
	sprintf(newname, "%s.Dimension", hard_devices[idx].dev_name);
	DEF(newname, DBL);	/* hard_devices[idx].dev_max_dim */
	sprintf(newname, "%s.OutputTitleFont", hard_devices[idx].dev_name);
	DEF(newname, STR);	/* hard_devices[idx].dev_title_font */
	sprintf(newname, "%s.OutputTitleSize", hard_devices[idx].dev_name);
	DEF(newname, DBL);	/* hard_devices[idx].dev_title_size */
	sprintf(newname, "%s.OutputAxisFont", hard_devices[idx].dev_name);
	DEF(newname, STR);	/* hard_devices[idx].dev_axis_font */
	sprintf(newname, "%s.OutputAxisSize", hard_devices[idx].dev_name);
	DEF(newname, DBL);	/* hard_devices[idx].dev_axis_size */
    }


    /* Read the default line and color attributes */
    for (idx = 0; idx < MAXATTR; idx++) {
	(void) sprintf(newname, "%d.Style", idx);
	DEF(newname, STYLE);	/* AllAttrs[idx].lineStyleLen */
	(void) sprintf(newname, "%d.Color", idx);
	DEF(newname, PIXEL);	/* AllAttrs[idx].pixelValue */
    }

    DEF("ReverseVideo", BOOL);
}


#define FS(str)	(void) fprintf(stderr, str)

static void
argerror(err, val)
char   *err,
       *val;
{
    (void) fprintf(stderr, "Error: %s: %s\n\n", val, err);

    FS("Usage: xgraph [-device <ps|X|hpgl|idraw|tgif>]\n");
    FS("\t[-bd border_color] [-bg background_color] [-fg foreground_color]\n");
    FS("\t[-bar] [-brb bar_base] [-brw bar_width] [-bof bar_offset] [-stk]\n");
    FS("\t[-bw bdr_width] [-db]  [-gw grid_size] [-fitx] [-fity]\n");
    FS("\t[-gs grid_style] [-lf label_font] [-lnx] [-lny] [-lw line_width]\n");
    FS("\t[-lx x1,x2] [-ly y1,y2] [-m] [-M] [-nl] [-ng] [-nb] [-p] [-P]\n");
    FS("\t[-rv] [-t title] [-tf title_font] [-tk] [-scale factor]\n");
    FS("\t[-x x_unit_name] [-y y_unit_name] [-fmtx format] [-fmty format]\n");
    FS("\t[[-geometry |=]W=H+X+Y] [[-display] <host>:<disp>.<screen>]\n");
    FS("\t[-Pprinter|-o output_file|-O output_file] [[-<digit> set_name]\n");
    FS("\t[-zg zero_color] [-zw zero_size] [-a] [-dl <delay>] input_files...\n\n");
    FS("-bar   Draw bar graph with base -brb, width -brw, and offset -bof\n");
    FS("-stk   Draw bar graph stacking data sets.\n");
    FS("-fitx  Scale all sets to fit the x-axis [0,1].\n");
    FS("-fity  Scale all sets to fit the y-axis [0,1].\n");
    FS("-fmtx  Printf format for the x-axis\n");
    FS("-fmty  Printf format for the y-axis\n");
    FS("-scale Scale the output file with factor\n");
    FS("-O fn  Printer ready output file\n");
    FS("-o fn  Encapsulated (document) output file\n");
    FS("-bb    Draw bounding box around data\n");
    FS("-db    Turn on debugging\n");
    FS("-lnx   Logarithmic scale for X axis\n");
    FS("-lny   Logarithmic scale for Y axis\n");
    FS("-m -M  Mark points distinctively (M varies with color)\n");
    FS("-nl    Don't draw lines (scatter plot)\n");
    FS("-ng    Don't draw legend\n");
    FS("-nb    Don't draw buttons\n");
    FS("-p -P  Mark points with dot (P means big dot)\n");
    FS("-rv    Reverse video on black and white displays\n");
    FS("-tk    Draw tick marks instead of full grid\n");
    FS("-a     Start in animation mode\n");
    FS("-dl    Animation delay.  Default is 2\n");

    exit(1);
}

#define ARG(opt, name) \
if (strcmp(argv[idx], opt) == 0) { \
    if (do_it) param_set(name, BOOL, "on"); \
    idx++; continue; \
}

#define ARG2(opt, name, type, missing) \
if (strcmp(argv[idx], opt) == 0) { \
   if (idx+1 >= argc) argerror(missing, argv[idx]); \
   if (do_it) param_set(name, type, argv[idx+1]); \
   idx += 2; continue;\
}

#define MAXLO	30

int
ParseArgs(argc, argv, do_it)
int     argc;
char   *argv[];
int     do_it;

/*
 * This routine parses the argument list for xgraph.  There are too
 * many to mention here so I won't.  If `do_it' is non-zero, options
 * are actually changed.  If `do_it' is zero, the argument list
 * is parsed but the options aren't set.  The routine is called
 * once to obtain the input files then again after the data is
 * read to set the options.
 */
{
    int     idx,
            set,
            dflag;
    char   *hi;

    dflag = DEF_OUTPUT_DEVICE;

    idx = 1;
    while (idx < argc) {
	if (argv[idx][0] == '-') {
	    /* Check to see if its a data set name */
	    if (sscanf(argv[idx], "-%d", &set) == 1) {
		/* The next string is a set name */
		if (idx + 1 >= argc)
		    argerror("missing set name", argv[idx]);
		if (do_it) {
		    PlotData[set].setName = argv[idx + 1];
		}
		idx += 2;
	    }
	    else {
		/* Some non-dataset option */
		ARG2("-x", "XUnitText", STR, "missing axis name");
		ARG2("-y", "YUnitText", STR, "missing axis name");
		ARG2("-t", "TitleText", STR, "missing plot title");
		ARG2("-fg", "Foreground", PIXEL, "missing color name");
		ARG2("-bg", "Background", PIXEL, "missing color name");
		ARG2("-bd", "Border", PIXEL, "missing color name");
		ARG2("-bw", "BorderSize", INT, "missing border size");
		ARG2("-zg", "ZeroColor", PIXEL, "missing color name");
		ARG2("-zw", "ZeroWidth", INT, "missing width");
		ARG2("-tf", "TitleFont", FONT, "missing font name");
		ARG2("-lf", "LabelFont", FONT, "missing font name");
                /* PW */
                ARG2("-dl", "DelayValue", INT, "missing delay value");
                /* Doesn't make much sense to PW why this must be
                   switched, but it must. */
                ARG2("-digy", "Format X", STR, "Missing C-String");
                ARG2("-digx", "Format Y", STR, "Missing C-String");
                ARG("-a", "Animate");
                /* End PW */
		ARG("-rv", "ReverseVideo");
		ARG("-tk", "Ticks");
		ARG("-tkax", "TickAxis");
		ARG("-bb", "BoundBox");
		if (strcmp(argv[idx], "-lx") == 0) {
		    /* Limit the X coordinates */
		    if (idx + 1 >= argc)
			argerror("missing coordinate(s)",
				 argv[idx]);
		    if (hi = index(argv[idx + 1], ',')) {
			char    low[MAXLO];

			(void) strncpy(low, argv[idx + 1], hi - argv[idx + 1]);
			low[hi - argv[idx + 1]] = '\0';
			hi++;
			if (do_it) {
			    param_set("XLowLimit", DBL, argv[idx + 1]);
			    param_set("XHighLimit", DBL, hi);
			}
		    }
		    else {
			argerror("limit coordinates not specified right",
				 argv[idx]);
		    }
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-ly") == 0) {
		    /* Limit the Y coordinates */
		    if (idx + 1 >= argc)
			argerror("missing coordinate(s)",
				 argv[idx]);
		    if (hi = index(argv[idx + 1], ',')) {
			char    low[MAXLO];

			(void) strncpy(low, argv[idx + 1], hi - argv[idx + 1]);
			low[hi - argv[idx + 1]] = '\0';
			hi++;
			if (do_it) {
			    param_set("YLowLimit", DBL, argv[idx + 1]);
			    param_set("YHighLimit", DBL, hi);
			}
		    }
		    else {
			argerror("limit coordinates not specified right",
				 argv[idx]);
		    }
		    idx += 2;
		    continue;
		}
		ARG2("-lw", "LineWidth", INT, "missing line width");
		ARG("-nl", "NoLines");
		ARG("-ng", "NoLegend");
		ARG("-nb", "NoButton");
		ARG("-m", "Markers");
		ARG("-M", "StyleMarkers");
		ARG("-p", "PixelMarkers");
		ARG("-P", "LargePixels");
		ARG("-lnx", "LogX");
		ARG("-lny", "LogY");
		ARG("-bar", "BarGraph");
		ARG("-stk", "StackGraph");
		ARG("-fitx", "FitX");
		ARG("-fity", "FitY");
		ARG2("-brw", "BarWidth", DBL, "missing width");
		ARG2("-bof", "BarOffset", DBL, "missing offset");
		ARG2("-brb", "BarBase", DBL, "missing base");
		ARG("-db", "Debug");
		ARG2("-gw", "GridSize", INT, "missing grid size");
		ARG2("-gs", "GridStyle", STYLE, "missing grid style");
		if (strcmp(argv[idx], "-display") == 0) {
		    /* Harmless display specification */
		    dflag = D_XWINDOWS;
		    disp_name = argv[idx+1];
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-geometry") == 0) {
		    if (do_it)
			param_set("Geometry", STR, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-device") == 0) {
		    if (idx + 1 >= argc)
			argerror("missing device", argv[idx]);
		    if (strcmp(argv[++idx], "hpgl") == 0)
			dflag = D_HPGL;
		    else if (strcmp(argv[idx], "idraw") == 0)
			dflag = D_IDRAW;
		    else if (strcmp(argv[idx], "x") == 0)
			dflag = D_XWINDOWS;
		    else if (strcmp(argv[idx], "ps") == 0)
			dflag = D_POSTSCRIPT;
		    else if (strcmp(argv[idx], "tgif") == 0)
			dflag = D_TGIF;
		    else
			argerror("bad device specification", argv[idx]);
		    idx++;
		    continue;
		}
		if (strncmp(argv[idx], "-P", 2) == 0) {
		    /* Printer spec */
		    if (do_it)
			param_set("Disposition", STR, "To Device");
		    if (do_it)
			param_set("FileOrDev", STR, &(argv[idx][2]));
		    idx++;
		    continue;
		}
		if (strcmp(argv[idx], "-o") == 0) {
		    if (do_it)
			param_set("Disposition", STR, "To File");
		    if (idx + 1 >= argc)
			argerror("missing file", argv[idx]);
		    if (do_it)
			param_set("FileOrDev", STR, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-O") == 0) {
		    if (do_it)
			param_set("Disposition", STR, "To File");
		    if (do_it)
			param_set("Document", BOOL, "on");
		    if (idx + 1 >= argc)
			argerror("missing file", argv[idx]);
		    if (do_it)
			param_set("FileOrDev", STR, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-fmtx") == 0) {
		    if (idx + 1 >= argc)
			argerror("missing x format", argv[idx]);
		    if (do_it)
			param_set("Format Y", STR, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-fmty") == 0) {
		    if (idx + 1 >= argc)
			argerror("missing y format", argv[idx]);
		    if (do_it)
			param_set("Format X", STR, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		if (strcmp(argv[idx], "-scale") == 0) {
		    if (idx + 1 >= argc)
			argerror("scale factor", argv[idx]);
		    if (do_it)
			param_set("Scale", DBL, argv[idx + 1]);
		    idx += 2;
		    continue;
		}
		argerror("unknown option", argv[idx]);
	    }
	}
	else if (argv[idx][0] == '=') {
	    /* Its a geometry specification */
	    if (do_it)
		param_set("Geometry", STR, argv[idx] + 1);
	    idx++;
	}
	else {
	    /* It might be the host:display string */
	    if (rindex(argv[idx], ':') == (char *) 0) {
		/* Should be an input file */
		inFileNames[numFiles] = argv[idx];
		numFiles++;
	    }
	    idx++;
	}
    }
    return (dflag);
}
