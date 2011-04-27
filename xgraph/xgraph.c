/* $Header$ */
/*
 * xgraph - A Simple Plotter for X
 *
 * David Harrison
 * University of California,  Berkeley
 * 1986, 1987, 1988, 1989
 *
 * Please see copyright.h concerning the formal reproduction rights
 * of this software.
 *
 * $Log$
 * Revision 1.2  2011-04-27 18:30:17  rlar
 * code cleanup
 *
 * Revision 1.1  2004/01/25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.3  1999/12/19 00:52:07  heideman
 * warning suppresion, slightly different flot ahndling
 *
 * Revision 1.2  1999/12/03 23:17:46  heideman
 * apply xgraph_no_animation.patch
 *
 * Revision 1.1.1.1  1999/12/03 23:15:52  heideman
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


extern void init_X();
extern void do_error();

#ifdef DO_DER
extern void Bounds();
#endif /* DO_DER */

static char *tildeExpand();
static void ReverseIt();
static void Traverse();
static int XErrHandler();


NewDataSet PlotData[MAXSETS],
           DataD1[MAXSETS],
           DataD2[MAXSETS];

XSegment *Xsegs[2];	/* Point space for X */

/* Basic transformation stuff */
double llx,
        lly,
        urx,
        ury;			/* Bounding box of all data */

static XContext win_context = (XContext) 0;

/* Other globally set defaults */

Display *disp;			/* Open display            */
Visual *vis;			/* Standard visual         */
Colormap cmap;			/* Standard colormap       */
int     screen;			/* Screen number           */
int     depth;			/* Depth of screen         */

int numFiles = 0;	/* Number of input files   */
char *inFileNames[MAXSETS];	/* File names              */

/* Total number of active windows */
int Num_Windows = 0;
char *Prog_Name;
char *disp_name;



main(argc, argv)
int     argc;
char   *argv[];

/*
 * This sets up the hard-wired defaults and reads the X defaults.
 * The command line format is: xgraph [host:display].
 */
{
    Window  primary,
            NewWindow();
    XEvent  theEvent;
    LocalWin *win_info;
    Cursor  zoomCursor;
    FILE   *strm;
    XColor  fg_color,
            bg_color;
    char    keys[MAXKEYS];
    int     nbytes,
            idx,
            maxitems = 0,
            flags;
    int     errs = 0;

    /* Open up new display */
    Prog_Name = argv[0];
    disp_name = "";

    /* Parse the argument list looking for input files */
    flags = ParseArgs(argc, argv, 0);

    if (flags == D_XWINDOWS) {
	disp = XOpenDisplay(disp_name);
	if (!disp) {
	    (void) fprintf(stderr,
			   "%s: cannot open display `%s'\n",
			   argv[0], disp_name);
	    exit(1);
	}
	XSetErrorHandler(XErrHandler);
    }

    /* Set up hard-wired defaults and allocate spaces */
    InitSets(flags);

    /* Read X defaults and override hard-coded defaults */
    if (PM_INT("Output Device") == D_XWINDOWS)
	ReadDefaults();


    /* Read the data into the data sets */
    llx = lly = MAXFLOAT;
    urx = ury = -MAXFLOAT;
    for (idx = 0; idx < numFiles; idx++) {
	strm = fopen(inFileNames[idx], "r");
	if (!strm) {
	    (void) fprintf(stderr, "Warning:  cannot open file `%s'\n",
			   inFileNames[idx]);
	}
	else {
	    if ((maxitems = ReadData(strm, inFileNames[idx])) < 0) {
		errs++;
	    }
	    (void) fclose(strm);
	}
    }
    if (!numFiles) {
	if ((maxitems = ReadData(stdin, (char *) 0)) < 0) {
	    errs++;
	}
    }
    if (errs) {
	(void) fprintf(stderr, "Problems found with input data.\n");
	exit(1);
    }

    /* Parse the argument list to set options */
    (void) ParseArgs(argc, argv, 1);
    if (PM_BOOL("Animate")) param_set("TitleText",STR,"Animated X Graph");

    if (maxitems == 0) {
	(void) fprintf(stderr, "Nothing to plot.\n");
	exit(1);
    }

    Xsegs[0] = (XSegment *) Malloc((unsigned) (maxitems * sizeof(XSegment)));
    Xsegs[1] = (XSegment *) Malloc((unsigned) (maxitems * sizeof(XSegment)));

    /* Reverse Video Hack */
    if (PM_BOOL("ReverseVideo"))
	ReverseIt();
    hard_init();
    if (PM_BOOL("Debug")) {
	if (PM_INT("Output Device") == D_XWINDOWS)
	    (void) XSynchronize(disp, 1);
	param_dump();
    }

    /* Logarithmic and bounding box computation */
    flags = 0;
    if (PM_BOOL("LogX"))
	flags |= LOG_X;
    if (PM_BOOL("LogY"))
	flags |= LOG_Y;
    if (PM_BOOL("StackGraph"))
	flags |= STK;
    if (PM_BOOL("FitX"))
	flags |= FITX;
    if (PM_BOOL("FitY"))
	flags |= FITY;
    Traverse(flags);

    /* Nasty hack here for bar graphs */
    if (PM_BOOL("BarGraph")) {
	double  base;

	llx -= PM_DBL("BarWidth");
	urx += PM_DBL("BarWidth");
	base = PM_DBL("BarBase");
	if (base < lly)
	    lly = base;
	if (base > ury)
	    ury = base;
    }

    /* Create initial window */
    if (PM_INT("Output Device") == D_XWINDOWS) {
        double asp;

        asp = 1.0;
	xtb_init(disp, screen, PM_PIXEL("Foreground"), PM_PIXEL("Background"),
		 PM_FONT("LabelFont"));
        
	primary = NewWindow(Prog_Name,
			    PM_DBL("XLowLimit"), PM_DBL("YLowLimit"),
			    PM_DBL("XHighLimit"), PM_DBL("YHighLimit"),
			    asp,0);
	if (!primary) {
	    (void) fprintf(stderr, "Main window would not open\n");
	    exit(1);
	}

	zoomCursor = XCreateFontCursor(disp, XC_sizing);
	fg_color = PM_COLOR("Foreground");
	bg_color = PM_COLOR("Background");
	XRecolorCursor(disp, zoomCursor, &fg_color, &bg_color);

	Num_Windows = 1;
	while (Num_Windows > 0) {
	    XNextEvent(disp, &theEvent);
	    if (xtb_dispatch(&theEvent) != XTB_NOTDEF)
		continue;
	    if (XFindContext(theEvent.xany.display,
			     theEvent.xany.window,
			     win_context, (caddr_t *) & win_info)) {
		/* Nothing found */
		continue;
	    }
	    switch (theEvent.type) {
	    case Expose:
		if (theEvent.xexpose.count <= 0) {
		    XWindowAttributes win_attr;

		    XGetWindowAttributes(disp, theEvent.xany.window, &win_attr);
		    win_info->dev_info.area_w = win_attr.width;
		    win_info->dev_info.area_h = win_attr.height;
		    init_X(win_info->dev_info.user_state);
                    EraseData(win_info);
		    DrawWindow(win_info);
		}
		break;
	    case KeyPress:
		nbytes = XLookupString(&theEvent.xkey, keys, MAXKEYS,
				       (KeySym *) 0, (XComposeStatus *) 0);
		for (idx = 0; idx < nbytes; idx++) {
		    if (keys[idx] == CONTROL_D) {
			/* Delete this window */
			DelWindow(theEvent.xkey.window, win_info);
		    }
		    else if (keys[idx] == CONTROL_C) {
			/* Exit program */
			Num_Windows = 0;
		    }
		    else if (keys[idx] == 'h') {
			PrintWindow(theEvent.xany.window, win_info);
		    }
		}
		break;
	    case ButtonPress:
		/* Handle creating a new window */
		Num_Windows += HandleZoom(Prog_Name,
					  &theEvent.xbutton,
					  win_info, zoomCursor);
		break;
	    default:
		(void) fprintf(stderr, "Unknown event type: %x\n",
			       theEvent.type);
		break;
	    }
	}
    }
    else {
	int     Device = PM_INT("Output Device");
	int     dflag = strcmp(PM_STR("Disposition"), "To Device") == 0;

	primary = NewWindow(Prog_Name,
			    PM_DBL("XLowLimit"), PM_DBL("YLowLimit"),
			    PM_DBL("XHighLimit"), PM_DBL("YHighLimit"),
			    1.0,0);
	do_hardcopy(Prog_Name, primary,
		    hard_devices[Device].dev_init,
		    dflag ? hard_devices[Device].dev_spec : 0,
		    PM_STR("FileOrDev"), (double) 19,
		    hard_devices[Device].dev_title_font,
		    hard_devices[Device].dev_title_size,
		    hard_devices[Device].dev_axis_font,
		    hard_devices[Device].dev_axis_size,
		    PM_BOOL("Document") * D_DOCU);
    }
    return 0;
}


#define BLACK_THRES	30000

static void
ReversePix(param_name)
char   *param_name;		/* Name of color parameter */

/*
 * Looks up `param_name' in the parameters database.  If found, the
 * color is examined and judged to be either black or white based
 * upon its red, green, and blue intensities.  The sense of the
 * color is then reversed and reset to its opposite.
 */
{
    params  val;

    if (param_get(param_name, &val)) {
	if ((val.pixv.value.red < BLACK_THRES) &&
	    (val.pixv.value.green < BLACK_THRES) &&
	    (val.pixv.value.blue < BLACK_THRES)) {
	    /* Color is black */
	    param_reset(param_name, "white");
	}
	else {
	    /* Color is white */
	    param_reset(param_name, "black");
	}
    }
    else {
	(void) fprintf(stderr, "Cannot reverse color `%s'\n", param_name);
    }
}

static void
ReverseIt()
/*
 * This routine attempts to implement reverse video.  It steps through
 * all of the important colors in the parameters database and makes
 * black white (and vice versa).
 */
{
    int     i;
    char    buf[1024];

    for (i = 0; i < MAXATTR; i++) {
	(void) sprintf(buf, "%d.Color", i);
	ReversePix(buf);
    }
    ReversePix("Foreground");
    ReversePix("Border");
    ReversePix("ZeroColor");
    ReversePix("Background");
}


static void
Traverse(flags)
int     flags;			/* Options */

/*
 * Traverses through all of the data applying certain options to the
 * data and computing the overall bounding box.  The flags are:
 *   LOG_X	Take the log of the X axis
 *   LOG_Y	Take the log of the Y axis
 *   STK	Stack coordinates.
 *   FITX	Fit x-coordinates from zero to one
 *   FITY	Fit y-coordinates from zero to one
 */
{
    int     i,
            j;
    PointList *spot;
    PointList *pspot;

    static char *paramstr[] =
    {
	"Cannot plot negative %s values\n",
	"when the logarithmic option is selected.\n",
	"Number of points in %d and %d don't match for stacking.\n",
	"Point %d in %d and %d doesn't match for stacking.\n",
	"Set %d has 0 %s.\n"
    };

    if (flags & (FITX|FITY)) 
	for (i = 0; i < MAXSETS; i++) 
	    for (spot = PlotData[i].list; spot; spot = spot->next) {
		float minx, maxx, miny, maxy;
		minx = maxx = spot->xvec[0];
		maxy = miny = spot->yvec[0];
		for (j = 1; j < spot->numPoints; j++) {
		    minx = MIN(minx, spot->xvec[j]);
		    miny = MIN(miny, spot->yvec[j]);
		    maxx = MAX(maxx, spot->xvec[j]);
		    maxy = MAX(maxy, spot->yvec[j]);
		}
		maxx = maxx - minx;
		maxy = maxy - miny;
		if (maxx == 0.0) {
		    (void) fprintf(stderr, paramstr[3], i, "width");
		    maxx = 1.0;
		}
		if (maxy == 0.0) {
		    (void) fprintf(stderr, paramstr[3], i, "height");
		    maxy = 1.0;
		}
		switch (flags & (FITX|FITY)) {
		case FITX:
		    for (j = 0; j < spot->numPoints; j++) 
			spot->xvec[j] = (-minx + spot->xvec[j]) / maxx;
		    break;
		case FITY:
		    for (j = 0; j < spot->numPoints; j++) 
			spot->yvec[j] = (-miny + spot->yvec[j]) / maxy;
		    break;
		case FITX|FITY:
		    for (j = 0; j < spot->numPoints; j++) {
			spot->xvec[j] = (-minx + spot->xvec[j]) / maxx;
			spot->yvec[j] = (-miny + spot->yvec[j]) / maxy;
		    }
		    break;
		default:
		    abort();
		}
	    }

    if (flags & STK)
	for (i = 1; i < MAXSETS; i++) {
	    for (spot = PlotData[i].list, pspot = PlotData[i - 1].list;
		 spot && pspot; spot = spot->next, pspot = pspot->next) {
		if (spot->numPoints != pspot->numPoints) {
		    (void) fprintf(stderr, paramstr[2], i - 1, i);
		    exit(1);
		}
		for (j = 0; j < spot->numPoints; j++) {
		    if (spot->xvec[j] != pspot->xvec[j]) {
			(void) fprintf(stderr, paramstr[3], j, i - 1, i);
			exit(1);
		    }
		    spot->yvec[j] += pspot->yvec[j];
		}
	    }
	}


    for (i = 0; i < MAXSETS; i++) {
	for (spot = PlotData[i].list; spot; spot = spot->next) {
	    for (j = 0; j < spot->numPoints; j++) {
		if (flags & LOG_Y) {
		    if (spot->yvec[j] > 0.0) {
			spot->yvec[j] = log10(spot->yvec[j]);
		    }
		    else if (spot->yvec[j] == 0)
			spot->yvec[j] = 0.0;
		    else {
			(void) fprintf(stderr, paramstr[0], "Y");
			(void) fprintf(stderr, paramstr[1]);
			exit(1);
		    }
		}
		if (flags & LOG_X) {
		    if (spot->xvec[j] > 0.0) {
			spot->xvec[j] = log10(spot->xvec[j]);
		    }
		    else if (spot->xvec[j] == 0)
			spot->xvec[j] = 0.0;
		    else {
			(void) fprintf(stderr, paramstr[0], "X");
			(void) fprintf(stderr, paramstr[1]);
			exit(1);
		    }
		}
		/* Update global bounding box */
		if (spot->xvec[j] < llx)
		    llx = spot->xvec[j];
		if (spot->xvec[j] > urx)
		    urx = spot->xvec[j];
		if (spot->yvec[j] < lly)
		    lly = spot->yvec[j];
		if (spot->yvec[j] > ury)
		    ury = spot->yvec[j];
	    }
	}
    }
}



/*
 * Button handling functions
 */

/*ARGSUSED*/
xtb_hret
del_func(win, bval, info)
Window  win;			/* Button window    */
int     bval;			/* Button value     */
char   *info;			/* User information */

/*
 * This routine is called when the `Close' button is pressed in
 * an xgraph window.  It causes the window to go away.
 */
{
    Window  the_win = (Window) info;
    LocalWin *win_info;

    xtb_bt_set(win, 1, (char *) 0, 0);
    if (!XFindContext(disp, the_win, win_context, (caddr_t *) & win_info)) {
	if (win_info->flags & HARDCOPY_IN_PROGRESS) {
	    do_error("Can't close window while\nhardcopy dialog is posted.\n");
	    xtb_bt_set(win, 0, (char *) 0, 0);
	}
	else {
	    DelWindow(the_win, win_info);
	}
    }
    return XTB_HANDLED;
}

/*ARGSUSED*/
xtb_hret
hcpy_func(win, bval, info)
Window  win;			/* Button Window    */
int     bval;			/* Button value     */
char   *info;			/* User Information */

/*
 * This routine is called when the hardcopy button is pressed
 * in an xgraph window.  It causes the output dialog to be
 * posted.
 */
{
    Window  the_win = (Window) info;
    LocalWin *win_info;

    xtb_bt_set(win, 1, (char *) 0, 0);
    if (!XFindContext(disp, the_win, win_context, (caddr_t *) & win_info)) {
	win_info->flags |= HARDCOPY_IN_PROGRESS;
	PrintWindow(the_win, win_info);
	win_info->flags &= (~HARDCOPY_IN_PROGRESS);
    }
    xtb_bt_set(win, 0, (char *) 0, 0);
    return XTB_HANDLED;
}

static
/*ARGSUSED*/
        xtb_hret
abt_func(win, bval, info)
Window  win;			/* Button window    */
int     bval;			/* Button value     */
char   *info;			/* User information */
{
    static char *msg_fmt =
    "Version %s\n\
XGraph + Animation and Derivatives\n\
Modification of code by David Harrison\n\
University of California, Berkeley\n\
(davidh@ic.Berkeley.EDU or \n\
...!ucbvax!ucbic!davidh)\n\
Animation, differentiation, and a few other\n\
new features added by Paul Walker,\n\
National Center for Supercomputer Applications\n\
and Univ. Illinois at U-C Dept of Physics.\n\
Send comments or suggestions to\n\
pwalker@ncsa.uiuc.edu\n";
    static int active = 0;
    char    msg_buf[1024];

    if (!active) {
	active = 1;
	xtb_bt_set(win, 1, (char *) 0, 0);
	(void) sprintf(msg_buf, msg_fmt, VERSION_STRING);
	msg_box("XGraph", msg_buf);
	xtb_bt_set(win, 0, (char *) 0, 0);
	active = 0;
    }
    return XTB_HANDLED;
}


#ifdef DO_DER
static
/*ARGSUSED  PW*/
        xtb_hret
rew_func(win, bval, info)
Window  win;			/* Button window    */
int     bval;			/* Button value     */
char   *info;			/* User information */
{
    /* This routine added, Paul Walker, to rewind the animation and start
       it over.  The only even moderatly tricky part is erasing the last
       item, which still lives, and redrawing the axis.  I do it by just
       copying the "Expose" information from the main routine. */
    Window  the_win = (Window) info;
    LocalWin *win_info;

    /* Set animation to True */
    param_set("Animate",BOOL,"on");

    if (!XFindContext(disp, the_win, win_context, (caddr_t *) & win_info)) {
	  XWindowAttributes win_attr;

	  XGetWindowAttributes(disp, the_win, &win_attr);
	  win_info->dev_info.area_w = win_attr.width;
	  win_info->dev_info.area_h = win_attr.height;
	  init_X(win_info->dev_info.user_state);
          EraseData (win_info);
	  DrawWindow(win_info);
    }

    return XTB_HANDLED;
}

static
/*ARGSUSED  PW*/
        xtb_hret
der_func(win, bval, info)
Window  win;			/* Button window    */
int     bval;			/* Button value     */
char   *info;			/* User information */
{
    /* This routine added, Paul Walker, to rewind the animation and start
       it over.  The only even moderatly tricky part is erasing the last
       item, which still lives, and redrawing the axis.  I do it by just
       copying the "Expose" information from the main routine. */
    Window  the_win = (Window) info;
    Window new_win;
    LocalWin *win_info;
    double loX,loY,hiX,hiY,asp;
    static char *msg_fmt =
    "Version %s\n\
Currently unable to display\n\
or calculate derivatives\n\
higher than 2nd order\n";
    static int active = 0;
    char    msg_buf[1024];

    XFindContext(disp, the_win, win_context, (caddr_t *) & win_info);
    if (win_info->DOrder == 2) {

      if (!active) {
	active = 1;
	xtb_bt_set(win, 1, (char *) 0, 0);
	(void) sprintf(msg_buf, msg_fmt, VERSION_STRING);
	msg_box("XGraph", msg_buf);
	xtb_bt_set(win, 0, (char *) 0, 0);
	active = 0;
      }
      return XTB_HANDLED;
    }
    Num_Windows += 1;
    asp = 1.0;
    Bounds(&loX,&loY,&hiX,&hiY,win_info->DOrder+1);
    new_win = NewWindow("Derivatives", loX, loY, hiX, hiY, asp, 
      win_info->DOrder+1);

    if (!XFindContext(disp, new_win, win_context, (caddr_t *) & win_info)) {
	  XWindowAttributes win_attr;

	  XGetWindowAttributes(disp, new_win, &win_attr);
	  win_info->dev_info.area_w = win_attr.width;
	  win_info->dev_info.area_h = win_attr.height;
	  init_X(win_info->dev_info.user_state);
          EraseData (win_info);
	  DrawWindow(win_info);
    }

    return XTB_HANDLED;
}


static
/*ARGSUSED  PW*/
        xtb_hret
rpl_func(win, bval, info)
Window  win;			/* Button window    */
int     bval;			/* Button value     */
char   *info;			/* User information */
{
    /* Puts us back into static mode ... */
    Window  the_win = (Window) info;
    LocalWin *win_info;

    /* Set animation to True */
    param_set("Animate",BOOL,"off");

    if (!XFindContext(disp, the_win, win_context, (caddr_t *) & win_info)) {
	  XWindowAttributes win_attr;

	  XGetWindowAttributes(disp, the_win, &win_attr);
	  win_info->dev_info.area_w = win_attr.width;
	  win_info->dev_info.area_h = win_attr.height;
	  init_X(win_info->dev_info.user_state);
          EraseData (win_info);
	  DrawWindow(win_info);
    }

    return XTB_HANDLED;
}
/* End change PW */

#endif /* DO_DER */

#define NORMSIZE	600
#define MINDIM		100

Window
NewWindow(progname, lowX, lowY, upX, upY, asp, DO)
char   *progname;		/* Name of program    */
double  lowX,
        lowY;			/* Lower left corner  */
double  upX,
        upY;			/* Upper right corner */
double  asp;			/* Aspect ratio       */
int     DO;                     /* Derivative Order.  */

/*
 * Creates and maps a new window.  This includes allocating its
 * local structure and associating it with the XId for the window.
 * The aspect ratio is specified as the ratio of width over height.
 */
{
    Window  new_window;
    LocalWin *new_info;
    static Cursor theCursor = (Cursor) 0;
    XSizeHints sizehints;
    XSetWindowAttributes wattr;
    XWMHints wmhints;
    XColor  fg_color,
            bg_color;
    int     geo_mask;
    int     width,
            height;
    unsigned long wamask;
    char    defSpec[120];
    double  pad;

    new_info = (LocalWin *) Malloc(sizeof(LocalWin));
    new_info->DOrder = DO;

    if (upX > lowX) {
	new_info->loX = lowX;
	new_info->hiX = upX;
    }
    else {
	new_info->loX = llx;
	new_info->hiX = urx;
    }
    if (upY > lowY) {
	new_info->loY = lowY;
	new_info->hiY = upY;
    }
    else {
	new_info->loY = lly;
	new_info->hiY = ury;
    }

    /* Increase the padding for aesthetics */
    if (new_info->hiX - new_info->loX == 0.0) {
	pad = MAX(0.5, fabs(new_info->hiX / 2.0));
	new_info->hiX += pad;
	new_info->loX -= pad;
    }
    if (new_info->hiY - new_info->loY == 0) {
	pad = MAX(0.5, fabs(ury / 2.0));
	new_info->hiY += pad;
	new_info->loY -= pad;
    }

    /* Add 10% padding to bounding box (div by 20 yeilds 5%) */
    pad = (new_info->hiX - new_info->loX) / 20.0;
    new_info->loX -= pad;
    new_info->hiX += pad;
    pad = (new_info->hiY - new_info->loY) / 20.0;
    new_info->loY -= pad;
    new_info->hiY += pad;

    /* Aspect ratio computation */
    if (asp < 1.0) {
	height = NORMSIZE;
	width = ((int) (((double) NORMSIZE) * asp));
    }
    else {
	width = NORMSIZE;
	height = ((int) (((double) NORMSIZE) / asp));
    }
    height = MAX(MINDIM, height);
    width = MAX(MINDIM, width);
    if (PM_INT("Output Device") == D_XWINDOWS) {
	(void) sprintf(defSpec, "%dx%d+100+100", width, height);

	wamask = CWBackPixel | CWBorderPixel | CWColormap;
	wattr.background_pixel = PM_PIXEL("Background");
	wattr.border_pixel = PM_PIXEL("Border");
	wattr.colormap = cmap;

	sizehints.flags = PPosition | PSize;
	sizehints.x = sizehints.y = 100;
	sizehints.width = width;
	sizehints.height = height;

	geo_mask = XParseGeometry(PM_STR("Geometry"), 
				  &sizehints.x, &sizehints.y,
				  (unsigned int *) &sizehints.width,
				  (unsigned int *) &sizehints.height);
	if (geo_mask & (XValue|YValue)) 
	    sizehints.flags = (sizehints.flags & ~PPosition) | USPosition;
	if (geo_mask & (WidthValue | HeightValue)) 
	    sizehints.flags = (sizehints.flags & ~PSize) | USSize;

	new_window = XCreateWindow(disp, RootWindow(disp, screen),
				   sizehints.x, sizehints.y,
				   (unsigned int) sizehints.width,
				   (unsigned int) sizehints.height,
				   (unsigned int) PM_INT("BorderSize"),
				   depth, InputOutput, vis,
				   wamask, &wattr);


	if (new_window) {
	    xtb_frame cl_frame,
	            hd_frame,
	            ab_frame,
                    rw_frame,
                    rp_frame,
                    dx_frame;

	    XStoreName(disp, new_window, progname);
	    XSetIconName(disp, new_window, progname);

	    wmhints.flags = InputHint | StateHint;
	    wmhints.input = True;
	    wmhints.initial_state = NormalState;
	    XSetWMHints(disp, new_window, &wmhints);

	    XSetWMNormalHints(disp, new_window, &sizehints);

	    /* Set device info */
	    set_X(new_window, &(new_info->dev_info));

	    if (!PM_BOOL("NoButton")) {
		/* Make buttons */
		xtb_bt_new(new_window, "Close", del_func,
			   (xtb_data) new_window, &cl_frame);
		new_info->close = cl_frame.win;
		XMoveWindow(disp, new_info->close, (int) BTNPAD, (int) BTNPAD);
		xtb_bt_new(new_window, "Hdcpy", hcpy_func,
			   (xtb_data) new_window, &hd_frame);
		new_info->hardcopy = hd_frame.win;
		XMoveWindow(disp, new_info->hardcopy,
			    (int) (BTNPAD + cl_frame.width + BTNINTER),
			    BTNPAD);
		xtb_bt_new(new_window, "About", abt_func,
			   (xtb_data) new_window, &ab_frame);
		new_info->about = ab_frame.win;
		XMoveWindow(disp, new_info->about,
			    (int) (BTNPAD + cl_frame.width + BTNINTER +
				   hd_frame.width + BTNINTER), BTNPAD);
#ifdef DO_DER
                /* These buttons added PW */
		xtb_bt_new(new_window, "Anim", rew_func,
		   (xtb_data) new_window, &rw_frame);
		new_info->rewind = rw_frame.win;
		XMoveWindow(disp, new_info->rewind,
		    (int) (BTNPAD + cl_frame.width + BTNINTER +
			   hd_frame.width + BTNINTER +
                           ab_frame.width + BTNINTER), BTNPAD);
		xtb_bt_new(new_window, "Replot", rpl_func,
		   (xtb_data) new_window, &rp_frame);
		new_info->replot = rp_frame.win;
		XMoveWindow(disp, new_info->replot,
		    (int) (BTNPAD + cl_frame.width + BTNINTER +
			   hd_frame.width + BTNINTER +
                           ab_frame.width + BTNINTER +
			   rw_frame.width + BTNINTER), BTNPAD);
		xtb_bt_new(new_window, "Deriv", der_func,
		   (xtb_data) new_window, &dx_frame);
		new_info->deriv = dx_frame.win;
		XMoveWindow(disp, new_info->deriv,
		    (int) (BTNPAD + cl_frame.width + BTNINTER +
			   hd_frame.width + BTNINTER +
                           ab_frame.width + BTNINTER +
			   rw_frame.width + BTNINTER +
			   rp_frame.width + BTNINTER), BTNPAD); 
#endif /* DO_DER */

		new_info->flags = 0;
	    }
	    XSelectInput(disp, new_window,
			 ExposureMask | KeyPressMask | ButtonPressMask);
	    if (!theCursor) {
		theCursor = XCreateFontCursor(disp, XC_top_left_arrow);
		fg_color = PM_COLOR("Foreground");
		bg_color = PM_COLOR("Background");
		XRecolorCursor(disp, theCursor, &fg_color, &bg_color);
	    }
	    XDefineCursor(disp, new_window, theCursor);
	    if (!win_context) {
		win_context = XUniqueContext();
	    }
	    XSaveContext(disp, new_window, win_context, (caddr_t) new_info);
	    XMapWindow(disp, new_window);
	    return new_window;
	}
	else {
	    return (Window) 0;
	}
    }
    else {
	new_info->dev_info.area_h = 1.0;
	new_info->dev_info.area_w = 1.0;
	return ((Window) new_info);
    }
}


DelWindow(win, win_info)
Window  win;			/* Window     */
LocalWin *win_info;		/* Local Info */

/*
 * This routine actually deletes the specified window and
 * decrements the window count.
 */
{
    xtb_data info;

    XDeleteContext(disp, win, win_context);
    xtb_bt_del(win_info->close, &info);
    xtb_bt_del(win_info->hardcopy, &info);
    xtb_bt_del(win_info->about, &info);
    Free((char *) win_info);
    XDestroyWindow(disp, win);
    Num_Windows -= 1;
}

PrintWindow(win, win_info)
Window  win;			/* Window       */
LocalWin *win_info;		/* Local Info   */

/*
 * This routine posts a dialog asking about the hardcopy
 * options desired.  If the user hits `OK',  the hard
 * copy is performed.
 */
{
    ho_dialog(win, Prog_Name, (char *) win_info);
}


static XRectangle boxEcho;
static GC echoGC = (GC) 0;

#define DRAWBOX \
if (startX < curX) { \
   boxEcho.x = startX; \
   boxEcho.width = curX - startX; \
} else { \
   boxEcho.x = curX; \
   boxEcho.width = startX - curX; \
} \
if (startY < curY) { \
   boxEcho.y = startY; \
   boxEcho.height = curY - startY; \
} else { \
   boxEcho.y = curY; \
   boxEcho.height = startY - curY; \
} \
XDrawRectangles(disp, win, echoGC, &boxEcho, 1);

#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)

#define TRANY(yval) \
(wi->UsrOppY - (((double) ((yval) - wi->XOrgY)) * wi->YUnitsPerPixel))


int
HandleZoom(progname, evt, wi, cur)
char   *progname;
XButtonPressedEvent *evt;
LocalWin *wi;
Cursor  cur;
{
    Window  win,
            new_win;
    Window  root_rtn,
            child_rtn;
    XEvent  theEvent;
    int     startX,
            startY,
            curX,
            curY,
            newX,
            newY,
            stopFlag,
            numwin = 0;
    int     root_x,
            root_y;
    unsigned int mask_rtn;
    double  loX,
            loY,
            hiX,
            hiY,
            asp;

    win = evt->window;
    if (XGrabPointer(disp, win, True,
		     (unsigned int) (ButtonPressMask | ButtonReleaseMask |
				   PointerMotionMask | PointerMotionHintMask),
		     GrabModeAsync, GrabModeAsync,
		     win, cur, CurrentTime) != GrabSuccess) {
	XBell(disp, 0);
	return 0;
    }
    if (echoGC == (GC) 0) {
	unsigned long gcmask;
	XGCValues gcvals;

	gcmask = GCForeground | GCFunction;
	gcvals.foreground = PM_PIXEL("ZeroColor") ^ PM_PIXEL("Background");
	gcvals.function = GXxor;
	echoGC = XCreateGC(disp, win, gcmask, &gcvals);
    }
    startX = evt->x;
    startY = evt->y;
    XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
		  &curX, &curY, &mask_rtn);
    /* Draw first box */
    DRAWBOX;
    stopFlag = 0;
    while (!stopFlag) {
	XNextEvent(disp, &theEvent);
	switch (theEvent.xany.type) {
	case MotionNotify:
	    XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
			  &newX, &newY, &mask_rtn);
	    /* Undraw the old one */
	    DRAWBOX;
	    /* Draw the new one */
	    curX = newX;
	    curY = newY;
	    DRAWBOX;
	    break;
	case ButtonRelease:
	    DRAWBOX;
	    XUngrabPointer(disp, CurrentTime);
	    stopFlag = 1;
	    if ((startX - curX != 0) && (startY - curY != 0)) {
		/* Figure out relative bounding box */
		loX = TRANX(startX);
		loY = TRANY(startY);
		hiX = TRANX(curX);
		hiY = TRANY(curY);
		if (loX > hiX) {
		    double  temp;

		    temp = hiX;
		    hiX = loX;
		    loX = temp;
		}
		if (loY > hiY) {
		    double  temp;

		    temp = hiY;
		    hiY = loY;
		    loY = temp;
		}
		/* physical aspect ratio */
		asp = ((double) ABS(startX - curX)) / 
		       ((double) ABS(startY - curY));
		new_win = NewWindow(progname, loX, loY, hiX, hiY, asp,
                  wi->DOrder);
		if (new_win) {
		    numwin = 1;
		}
		else {
		    numwin = 0;
		}
	    }
	    else {
		numwin = 0;
	    }
	    break;
	default:
	    printf("unknown event: %d\n", theEvent.xany.type);
	    break;
	}
    }
    return numwin;
}


#define RND(val)	((int) ((val) + 0.5))

/*ARGSUSED*/
void
do_hardcopy(prog, info, init_fun, dev_spec, file_or_dev, maxdim,
	    ti_fam, ti_size, ax_fam, ax_size, doc_p)
char   *prog;			/* Program name for Xdefaults    */
char   *info;			/* Some state information        */
int     (*init_fun) ();		/* Hardcopy init function        */
char   *dev_spec;		/* Device specification (if any) */
char   *file_or_dev;		/* Filename or device spec       */
double  maxdim;			/* Maximum dimension in cm       */
char   *ti_fam,
       *ax_fam;			/* Font family names             */
double  ti_size,
        ax_size;		/* Font sizes in points          */
int     doc_p;			/* Documentation predicate       */

/*
 * This routine resets the function pointers to those specified
 * by `init_fun' and causes a screen redisplay.  If `dev_spec'
 * is non-zero,  it will be considered a sprintf string with
 * one %s which will be filled in with `file_or_dev' and fed
 * to popen(3) to obtain a stream.  Otherwise,  `file_or_dev'
 * is considered to be a file and is opened for writing.  The
 * resulting stream is fed to the initialization routine for
 * the device.
 */
{
    LocalWin *curWin = (LocalWin *) info;
    LocalWin thisWin;
    FILE   *out_stream;
    char    buf[MAXBUFSIZE],
            err[MAXBUFSIZE],
            ierr[ERRBUFSIZE];
    char    tilde[MAXBUFSIZE * 10];
    int     final_w,
            final_h,
            flags;
    double  ratio;

    if (dev_spec) {
	(void) sprintf(buf, dev_spec, file_or_dev);
	out_stream = popen(buf, "w");
	if (!out_stream) {
	    sprintf(err, "Unable to issue command:\n  %s\n", buf);
	    do_error(err);
	    return;
	}
    }
    else {
	tildeExpand(tilde, file_or_dev);
	out_stream = fopen(tilde, "w");
	if (!out_stream) {
	    sprintf(err, "Unable to open file `%s'\n", tilde);
	    do_error(err);
	    return;
	}
    }
    if (curWin != (LocalWin *) 0) {
	thisWin = *curWin;
	ratio = ((double) thisWin.dev_info.area_w) /
	    ((double) thisWin.dev_info.area_h);
    }
    else
	ratio = 1.0;

    if (thisWin.dev_info.area_w > thisWin.dev_info.area_h) {
	final_w = RND(maxdim * 10000.0 * PM_DBL("Scale"));
	final_h = RND(maxdim / ratio * 10000.0 * PM_DBL("Scale"));
    }
    else {
	final_w = RND(maxdim * ratio * 10000.0 * PM_DBL("Scale"));
	final_h = RND(maxdim * 10000.0 * PM_DBL("Scale"));
    }
    ierr[0] = '\0';
    flags = 0;
    if (doc_p)
	flags |= D_DOCU;
    if (init_fun (out_stream, final_w, final_h, ti_fam, ti_size,
		     ax_fam, ax_size, flags, &(thisWin.dev_info), ierr)) {
	DrawWindow(&thisWin);
	if (thisWin.dev_info.xg_end) {
	    thisWin.dev_info.xg_end(thisWin.dev_info.user_state);
	}
    }
    else {
	do_error(ierr);
    }
    if (dev_spec) {
	(void) pclose(out_stream);
    }
    else {
	(void) fclose(out_stream);
    }
}


static char *
tildeExpand(out, in)
char   *out;			/* Output space for expanded file name */
char   *in;			/* Filename with tilde                 */

/*
 * This routine expands out a file name passed in `in' and places
 * the expanded version in `out'.  It returns `out'.
 */
{
    char    username[50],
           *userPntr;
    struct passwd *userRecord;

    out[0] = '\0';

    /* Skip over the white space in the initial path */
    while ((*in == ' ') ||(*in == '\t'))
	in    ++;

    /* Tilde? */
    if (in[0] == TILDE) {
	/* Copy user name into 'username' */
	in    ++;

	userPntr = &(username[0]);
	while ((*in !='\0') &&(*in !='/')) {
	    *(userPntr++) = *(in ++);
	}
	*(userPntr) = '\0';
	/* See if we have to fill in the user name ourselves */
	if (strlen(username) == 0) {
	    userRecord = getpwuid(getuid());
	}
	else {
	    userRecord = getpwnam(username);
	}
	if (userRecord) {
	    /* Found user in passwd file.  Concatenate user directory */
	    strcat(out, userRecord->pw_dir);
	}
    }

    /* Concantenate remaining portion of file name */
    strcat(out, in);
    return out;
}



#define ERR_MSG_SIZE	2048

/*ARGSUSED*/
static int
XErrHandler(disp_ptr, evt)
Display *disp_ptr;
XErrorEvent *evt;

/*
 * Displays a nicely formatted message and core dumps.
 */
{
    char    err_buf[ERR_MSG_SIZE],
            mesg[ERR_MSG_SIZE],
            number[ERR_MSG_SIZE];
    char   *mtype = "XlibMessage";

    XGetErrorText(disp_ptr, evt->error_code, err_buf, ERR_MSG_SIZE);
    (void) fprintf(stderr, "X Error: %s\n", err_buf);
    XGetErrorDatabaseText(disp_ptr, mtype, "MajorCode",
			  "Request Major code %d", mesg, ERR_MSG_SIZE);
    (void) fprintf(stderr, mesg, evt->request_code);
    (void) sprintf(number, "%d", evt->request_code);
    XGetErrorDatabaseText(disp_ptr, "XRequest", number, "", err_buf,
			  ERR_MSG_SIZE);
    (void) fprintf(stderr, " (%s)\n", err_buf);

    abort();
}
