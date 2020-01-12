/* Copyright: Holger Vogt, 2020 */
/* Three Clause BSD */
/* Universal color table and retrival */


#include "ngspice/ngspice.h"

#ifdef HAS_WINGUI

#include "ngspice/cpextern.h"
#include "ngspice/hash.h"
#include "ngspice/macros.h"
#undef BOOLEAN
#include <windows.h>
#include "ngspice/wincolornames.h"



static NGHASHPTR color_p; /* color hash table */
static COLORREF get_wincolor(char* name, int nocolor);

void wincolor_init_hash(COLORREF *ColorTable, int noc)
{
    int i;
    char buf[BSIZE_SP], colorstring[BSIZE_SP];
    int nocolor = NUMELEMS(ctable);
    color_p = nghash_init(nocolor);
    nghash_unique(color_p, FALSE);
    for (i = 0; i < nocolor; i++) {
        strtolower(ctable[i].name);
        ctable[i].rgbc = RGB(ctable[i].R, ctable[i].G, ctable[i].B);
        nghash_insert(color_p, ctable[i].name, &(ctable[i].rgbc));
    }

    for (i = 0; i < noc; i++) {
        (void) sprintf(buf, "color%d", i);
        if (!cp_getvar(buf, CP_STRING, colorstring, sizeof(colorstring)))
            (void) strcpy(colorstring, stdcolornames[i]);
        COLORREF* val = (COLORREF*)nghash_find(color_p, colorstring);
        if(val)
            ColorTable[i] = *val;
        else
            ColorTable[i] = 0; 
    }
}


/* ColorTable[0]: background, ColorTable[1]: grid, text */
void wincolor_init(COLORREF* ColorTable, int noc)
{
    int i;
    static bool bgisblack = TRUE;
    char buf[BSIZE_SP], colorstring[BSIZE_SP];
    int nocolor = NUMELEMS(ctable);

    for (i = 0; i < nocolor; i++) {
        strtolower(ctable[i].name);
        ctable[i].rgbc = RGB(ctable[i].R, ctable[i].G, ctable[i].B);
    }
    i = 0;
    while(i < noc) {
        /* when color0 is set to white and color1 is not given, set ColorTable[2] to black */
        (void)sprintf(buf, "color%d", i);
        if (!cp_getvar(buf, CP_STRING, colorstring, sizeof(colorstring))) {
            if (i == 1) {
                /* switch the grid and text color depending on background */
                int tcolor = GetRValue(ColorTable[0]) + GetGValue(ColorTable[0]) + GetBValue(ColorTable[0]);
                if (tcolor > 250) {
                    ColorTable[1] = RGB(0, 0, 0);
                    i++;
                    bgisblack = FALSE;
                    continue;
                }
                else {
                    ColorTable[1] = RGB(255, 255, 255);
                    i++;
                    bgisblack = TRUE;
                    continue;
                }
            }
            /* old code: beginning with 12 the colors are repeated */
            else if (!bgisblack && (i == 12))
                (void)strcpy(colorstring, "black");
            else
                (void)strcpy(colorstring, stdcolornames[i]);
        }
        ColorTable[i] =  get_wincolor(colorstring, nocolor);
        i++;
    }
}

void wincolor_redo(COLORREF* ColorTable, int noc)
{
    int i = 0;
    static bool bgisblack = TRUE;
    char buf[BSIZE_SP], colorstring[BSIZE_SP];
    int nocolor = NUMELEMS(ctable);

    while (i < noc) {
        /* when color0 is set to white and color1 is not given, set ColorTable[2] to black */
        (void)sprintf(buf, "color%d", i);
        if (!cp_getvar(buf, CP_STRING, colorstring, sizeof(colorstring))) {
            if (i == 1) {
                /* switch the grid and text color depending on background */
                int tcolor = GetRValue(ColorTable[0]) + 
                    (int)(1.5 * GetGValue(ColorTable[0])) + GetBValue(ColorTable[0]);
                if (tcolor > 360) {
                    ColorTable[1] = RGB(0, 0, 0);
                    i++;
                    bgisblack = FALSE;
                    continue;
                }
                else {
                    ColorTable[1] = RGB(255, 255, 255);
                    i++;
                    bgisblack = TRUE;
                    continue;
                }
            }
            /* old code: beginning with 12 the colors are repeated */
            else if (!bgisblack && (i == 12))
                (void)strcpy(colorstring, "black");
            else
                (void)strcpy(colorstring, stdcolornames[i]);
        }
        ColorTable[i] = get_wincolor(colorstring, nocolor);
        i++;
    }
}

COLORREF *get_wincolor_hash(char *name)
{
    return nghash_find(color_p, name);
}

static COLORREF get_wincolor(char *name, int nocolor)
{
    
    int i;
    for (i = 0; i < nocolor; i++) {
        if (ciprefix(name, ctable[i].name)) {
            return ctable[i].rgbc;
        }
    }
    fprintf(stderr, "Warning: Color %s is not available\n", name);
    fprintf(stderr, "    Color 'green' is returned instead!\n");
    return RGB(0, 255, 0);
}

#endif
