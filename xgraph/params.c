/* $Header$ */
/*
 * Xgraph Parameters
 *
 * This file contains routines for setting and retrieving
 * various X style display parameters for xgraph.
 *
 * $Log$
 * Revision 1.1  2004-01-25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.2  1999/12/08 19:32:41  heideman
 * strcasecmp portability fix
 *
 * Revision 1.1.1.1  1999/12/03 23:15:52  heideman
 * xgraph-12.0
 *
 */
#ifndef lint
static char rcsid[] = "$Id$";

#endif

#include <ctype.h>
#include <stdio.h>
#include "st.h"
#include "params.h"
#include "hard_devices.h"
#include "xgraph.h" /* for string.h */

/* For use by convenience macros */
params  param_temp,
       *param_temp_ptr;
XColor  param_null_color =
{0, 0, 0, 0, 0, 0};
param_style param_null_style =
{STYLE, 0, (char *) 0};

static st_table *param_table = (st_table *) 0;

typedef struct param_full_defn {
    param_types type;
    char   *text_form;
    params *real_form;
}       param_full;


#define ISO_FONT "*-*-%s-medium-r-normal-*-*-%d-*-*-*-*-iso8859-*"
static Display *param_disp;
static Colormap param_cmap;
static int param_scrn;

static void free_resource();
static params *resolve_entry();
static int strihash();
static int do_color();
static int do_font();
static int do_style();
static int do_bool();

#define DEF_INT		"0"
#define DEF_STR		""
#define DEF_FONT	"fixed"
#define DEF_PIXEL	"black"
#define DEF_STYLE	"1"
#define DEF_BOOL	"false"
#define DEF_DBL		"0.0"

#define DEF_MAX_FONT	1024
#define DEF_MAX_NAMES	10

#define DUP(str)	\
strcpy((char *) Malloc((unsigned) (strlen(str)+1)), (str))


void
param_init(disp, cmap)
Display *disp;			/* X Connection        */
Colormap cmap;			/* Colormap for colors */

/*
 * Initializes parameter package.  The display and colormap arguments
 * are used to resolve font and pixel values.
 */
{
    param_table = st_init_table(stricmp, strihash);
    if (disp != NULL) {
	param_disp = disp;
	param_cmap = cmap;
	/* This could also be a parameter for greater generality */
	param_scrn = DefaultScreen(disp);
    }
}



void
param_set(name, type, val)
char   *name;			/* Name of parameter   */
param_types type;		/* Type                */
char   *val;			/* Text form for value */

/*
 * Sets the parameter with the given name to have the type
 * `type' and the text value `value'.  This will be evaluated
 * to its full form the first time it is referenced using
 * param_get().  If it is already filled, the old value
 * will be reclaimed.
 */
{
    param_full *entry;

    if (!param_table) {
	(void) fprintf(stderr, "Parameter table not initialized\n");
	return;
    }
    if (st_lookup(param_table, name, (char **) &entry)) {
	if (entry->real_form)
	    free_resource(entry->real_form);
	entry->real_form = (params *) 0;
    }
    else {
	entry = (param_full *) Malloc(sizeof(param_full));
	entry->text_form = (char *) 0;
	entry->real_form = (params *) 0;
	(void) st_insert(param_table, DUP(name), (char *) entry);
    }
    entry->type = type;
    if (entry->text_form)
	(void) Free((char *) (entry->text_form));
    entry->text_form = DUP(val);
}


void
param_reset(name, val)
char   *name;			/* Name of parameter   */
char   *val;			/* Text form for value */

/*
 * This routine sets the value of an existing parameter to a new
 * value.  The type of the parameter remains the same.  Changes
 * in type should be done by using param_set() directly.
 */
{
    param_full *entry;

    if (!param_table) {
	(void) fprintf(stderr, "Parameter table not initialized\n");
	return;
    }
    if (st_lookup(param_table, name, (char **) &entry))
	param_set(name, entry->type, val);
    else
	(void) fprintf(stderr, "Cannot reset unknown parameter `%s'\n", name);
}




params *
param_get(name, val)
char   *name;			/* Name of parameter */
params *val;			/* Result value      */

/*
 * Retrieves a value from the parameter table.  The value
 * is placed in `val'.  If successful, the routine will
 * return `val'.  Otherwise, it will return zero.
 */
{
    param_full *entry;

    if (!param_table) {
	(void) fprintf(stderr, "Parameter table not initialized\n");
	return (params *) 0;
    }
    if (st_lookup(param_table, name, (char **) &entry)) {
	if (!entry->real_form)
	    entry->real_form = resolve_entry(name, entry->type,
					     entry->text_form);
	*val = *(entry->real_form);
	return val;
    }
    else {
	return (params *) 0;
    }
}


static void
free_resource(val)
params *val;			/* Value to free */

/*
 * Reclaims a resource based on its type.
 */
{
    switch (val->type) {
    case INT:
    case STR:
    case BOOL:
    case DBL:
	/* No reclaimation necessary */
	break;
    case PIXEL:
	if ((val->pixv.value.pixel != WhitePixel(param_disp, param_scrn)) &&
	    (val->pixv.value.pixel != BlackPixel(param_disp, param_scrn)))
	    XFreeColors(param_disp, param_cmap, &(val->pixv.value.pixel), 1, 0);
	break;
    case FONT:
	XFreeFont(param_disp, val->fontv.value);
	break;
    case STYLE:
	(void) Free(val->stylev.dash_list);
	break;
    }
    (void) Free((char *) val);
}



static params *
resolve_entry(name, type, form)
char   *name;			/* Name of item for errors */
param_types type;		/* What type of thing */
char   *form;			/* Textual form       */

/*
 * Allocates and returns an appropriate parameter structure
 * by translating `form' into its native type as given by `type'.
 * If it can't translate the given form, it will fall back onto
 * the default.
 */
{
    static char paramstr[] =
    "Parameter %s: can't translate `%s' into a %s (defaulting to `%s')\n";
    params *result = (params *) Malloc(sizeof(params));

    result->type = type;
    switch (type) {
    case INT:
	if (sscanf(form, "%d", &result->intv.value) != 1) {
	    (void) fprintf(stderr, paramstr, name, form, "integer", DEF_INT);
	    result->intv.value = atoi(DEF_INT);
	}
	break;
    case STR:
	result->strv.value = form;
	break;
    case PIXEL:
	if (!do_color(form, &result->pixv.value)) {
	    (void) fprintf(stderr, paramstr, name, form, "color", DEF_PIXEL);
	    (void) do_color(DEF_PIXEL, &result->pixv.value);
	}
	break;
    case FONT:
	if (!do_font(form, &result->fontv.value)) {
	    (void) fprintf(stderr, paramstr, name, form, "font", DEF_FONT);
	    (void) do_font(DEF_FONT, &result->fontv.value);
	}
	break;
    case STYLE:
	if (!do_style(form, &result->stylev)) {
	    (void) fprintf(stderr, paramstr, name, form, "line style",
			   DEF_STYLE);
	    (void) do_style(DEF_STYLE, &result->stylev);
	}
	break;
    case BOOL:
	if (!do_bool(form, &result->boolv.value)) {
	    (void) fprintf(stderr, paramstr, name, form, "boolean flag",
			   DEF_BOOL);
	    (void) do_bool(DEF_BOOL, &result->boolv.value);
	}
	break;
    case DBL:
	if (sscanf(form, "%lf", &result->dblv.value) != 1) {
	    (void) fprintf(stderr, paramstr, name, form, "double", DEF_DBL);
	    result->dblv.value = atof(DEF_DBL);
	}
	break;
    }
    return result;
}



static int
do_color(name, color)
char   *name;			/* Name for color */
XColor *color;			/* Returned color */

/*
 * Translates `name' into a color and attempts to get the pixel
 * for the color using XAllocColor().
 */
{
    int     result = 1;

    if (PM_INT("Output Device") == D_XWINDOWS) {
	if (XParseColor(param_disp, param_cmap, name, color)) {
	    if (stricmp(name, "black") == 0) {
		color->pixel = BlackPixel(param_disp, param_scrn);
		XQueryColor(param_disp, param_cmap, color);
	    }
	    else if (stricmp(name, "white") == 0) {
		color->pixel = WhitePixel(param_disp, param_scrn);
		XQueryColor(param_disp, param_cmap, color);
	    }
	    else
		result = XAllocColor(param_disp, param_cmap, color);
	}
	else
	    result = 0;
    }
    return result;
}



static int
do_font(name, font_info)
char   *name;			/* Name of desired font      */
XFontStruct **font_info;	/* Returned font information */

/*
 * This routine translates a font name into a font structure.  The
 * font name can be in two forms.  The first form is <family>-<size>.
 * The family is a family name (like helvetica) and the size is
 * in points (like 12).  If the font is not in this form, it
 * is assumed to be a regular X font name specification and
 * is looked up using the standard means.
 */
{
    char    name_copy[DEF_MAX_FONT],
            query_spec[DEF_MAX_FONT];
    char   *font_family,
           *font_size,
          **font_list;
    int     font_size_value,
            font_count,
            i;

    /* First attempt to interpret as font family/size */
    if (PM_INT("Output Device") == D_XWINDOWS) {
	(void) strcpy(name_copy, name);
	if (font_size = index(name_copy, '-')) {
	    *font_size = '\0';
	    font_family = name_copy;
	    font_size++;
	    font_size_value = atoi(font_size);
	    if (font_size_value > 0) {
		/*
		 * Still a little iffy -- what about weight and roman vs. other
		 */
		(void) sprintf(query_spec, ISO_FONT,
			       font_family, font_size_value * 10);
		font_list = XListFonts(param_disp, query_spec,
				       DEF_MAX_NAMES, &font_count);

		/* Load first one that you can */
		for (i = 0; i < font_count; i++)
		    if (*font_info = XLoadQueryFont(param_disp, font_list[i]))
			break;
		if (*font_info)
		    return 1;
	    }
	}
	/* Assume normal font name */
	return (int) (*font_info = XLoadQueryFont(param_disp, name));
    }
}


static int
do_style(list, val)
char   *list;			/* List of ones and zeros */
param_style *val;		/* Line style returned    */

/*
 * Translates a string representation of a dash specification into
 * a form suitable for use in XSetDashes().  Assumes `list'
 * is a null terminated string of ones and zeros.
 */
{
    char   *i,
           *spot,
            last_char;
    int     count;

    for (i = list; *i; i++)
	if ((*i != '0') && (*i != '1'))
	    break;

    if (!*i) {
	val->len = 0;
	last_char = '\0';
	for (i = list; *i; i++) {
	    if (*i != last_char) {
		val->len += 1;
		last_char = *i;
	    }
	}
	val->dash_list = (char *) Malloc((unsigned)
					 (sizeof(char) * val->len + 1));
	last_char = *list;
	spot = val->dash_list;
	count = 0;
	for (i = list; *i; i++) {
	    if (*i != last_char) {
		*spot++ = (char) count;
		last_char = *i;
		count = 1;
	    }
	    else
		count++;
	}
	*spot = (char) count;
	return 1;
    }
    else {
	return 0;
    }
}


static char *positive[] =
{"on", "yes", "true", "1", "affirmative", (char *) 0};
static char *negative[] =
{"off", "no", "false", "0", "negative", (char *) 0};

static int
do_bool(name, val)
char   *name;			/* String representation */
int    *val;			/* Returned value        */

/*
 * Translates a string representation into a suitable binary value.
 * Can parse all kinds of interesting boolean type words.
 */
{
    char  **term;

    for (term = positive; *term; term++) {
	if (stricmp(name, *term) == 0)
	    break;
    }
    if (*term) {
	*val = 1;
	return 1;
    }
    for (term = negative; *term; term++)
	if (stricmp(name, *term) == 0)
	    break;

    if (*term) {
	*val = 0;
	return 1;
    }
    return 0;
}



/*ARGSUSED*/
static enum st_retval
dump_it(key, value, arg)
char   *key,
       *value,
       *arg;
{
    param_full *val = (param_full *) value;

    (void) fprintf(stdout, "%s (", key);
    switch (val->type) {
    case INT:
	(void) fprintf(stdout, "INT");
	break;
    case STR:
	(void) fprintf(stdout, "STR");
	break;
    case PIXEL:
	(void) fprintf(stdout, "PIXEL");
	break;
    case FONT:
	(void) fprintf(stdout, "FONT");
	break;
    case STYLE:
	(void) fprintf(stdout, "STYLE");
	break;
    case BOOL:
	(void) fprintf(stdout, "BOOL");
	break;
    case DBL:
	(void) fprintf(stdout, "DBL");
	break;
    }
    (void) fprintf(stdout, ") = %s\n", val->text_form);
    return ST_CONTINUE;
}

void
param_dump()
/*
 * Dumps all of the parameter values to standard output.
 */
{
    st_foreach(param_table, dump_it, (char *) 0);
}



#ifdef HAVE_STRCASECMP
int
stricmp(a, b)
	char *a, *b;
{
	return strcasecmp(a, b);
}
#else
int
stricmp(a, b)
register char *a,
       *b;

/*
 * This routine compares two strings disregarding case.
 */
{
    register int value;

    if ((a == (char *) 0) || (b == (char *) 0)) {
	return a - b;
    }

    for ( /* nothing */ ;
	 ((*a | *b) &&
	  !(value = ((isupper(*a) ? *a - 'A' + 'a' : *a) -
		     (isupper(*b) ? *b - 'A' + 'a' : *b))));
	 a++, b++)
	 /* Empty Body */ ;

    return value;
}

#endif

static int
strihash(string, modulus)
register char *string;
int     modulus;

/* Case insensitive computation */
{
    register int val = 0;
    register int c;

    while ((c = *string++) != '\0') {
	if (isupper(c))
	    c = tolower(c);
	val = val * 997 + c;
    }

    return ((val < 0) ? -val : val) % modulus;
}
