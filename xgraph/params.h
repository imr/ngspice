/*
 * Xgraph parameters
 */

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "xgraph.h"

/* If you have an ANSI compiler,  some checking will be done */
#ifdef __STDC__
#define DECLARE(func, rtn, args)	extern rtn func args
#else
#define DECLARE(func, rtn, args)	extern rtn func ()
#endif

typedef enum param_types_defn {
    INT, STR, PIXEL, FONT, STYLE, BOOL, DBL
}       param_types;

typedef struct params_int_defn {
    param_types type;		/* INT */
    int     value;
}       param_int;

typedef struct params_str_defn {
    param_types type;		/* STR */
    char   *value;
}       param_str;

typedef struct params_pix_defn {
    param_types type;		/* PIXEL */
    XColor  value;
}       param_pix;

typedef struct params_font_defn {
    param_types type;		/* FONT */
    XFontStruct *value;
}       param_font;

typedef struct params_style_defn {
    param_types type;		/* STYLE */
    int     len;
    char   *dash_list;
}       param_style;

typedef struct params_bool_defn {
    param_types type;		/* BOOL */
    int     value;
}       param_bool;

typedef struct params_dbl_defn {
    param_types type;		/* DBL */
    double  value;
}       param_dbl;

typedef union params_defn {
    param_types type;
    param_int intv;		/* INT */
    param_str strv;		/* STR */
    param_pix pixv;		/* PIXEL */
    param_font fontv;		/* FONT */
    param_style stylev;		/* STYLE */
    param_bool boolv;		/* BOOL */
    param_dbl dblv;		/* DBL */
}       params;

DECLARE(param_init, void, (Display * disp, Colormap cmap));
DECLARE(param_set, void, (char *name, param_types type, char *val));
DECLARE(param_reset, void, (char *name, char *val));
DECLARE(param_get, params *, (char *name, params * val));
DECLARE(param_dump, void, ());

#ifdef stricmp
#undef stricmp
#endif
DECLARE(stricmp, int, (char *a, char *b));

/* Some convenience macros */

extern params param_temp,
       *param_temp_ptr;
extern XColor param_null_color;
extern param_style param_null_style;

#define PM_INT(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->intv.value : \
 (abort(), (int) 0))

#define PM_STR(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->strv.value : \
 (abort(), (char *) 0))

#define PM_COLOR(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->pixv.value : \
 (abort(), param_null_color))

#define PM_PIXEL(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->pixv.value.pixel : \
 (abort(), (Pixel) 0))

#define PM_FONT(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->fontv.value : \
 (abort(), (XFontStruct *) 0))

#define PM_STYLE(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->stylev : \
 (abort(), param_null_style))

#define PM_BOOL(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->boolv.value : \
 (abort(), 0))

#define PM_DBL(name)	\
((param_temp_ptr = param_get(name, &param_temp)) ? \
 param_temp_ptr->dblv.value : \
 (abort(), 0.0))


#endif				/* _PARAMS_H_ */
