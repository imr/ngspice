/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Read and write the ascii and binary rawfile formats.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "dimens.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"
#include "ngspice/stringskip.h"

#include "rawfile.h"
#include "variable.h"
#include "../misc/misc_time.h"

#include "ngspice/compatmode.h"


static void fixdims(struct dvec *v, char *s);


int raw_prec = -1;        /* How many sigfigs to use, default 15 (max).  */

#define DEFPREC 15

#ifdef HAS_WINGUI
#undef fscanf             /* redo I/O from WINMAIN.C here
                             otherwise reading ASCII will not work */
#endif


/* Write a raw file with the 'write' command.  We write everything in the plot pointed to.
   Writing a raw file in batch mode is handled by fileInit_pass1() and fileInit_pass2()
   in outitf.c */
void raw_write(char *name, struct plot *pl, bool app, bool binary)
{
    FILE *fp;
    bool realflag = TRUE, writedims;
    bool raw_padding;
    int length, numdims, dims[MAXDIMS];
    int nvars, i, j, prec;
    struct dvec *v, *lv;
    wordlist *wl;
    struct variable *vv;
    double dd;
    char buf[BSIZE_SP];
    char *branch;
    bool keepbranch = FALSE;

    raw_padding = !cp_getvar("nopadding", CP_BOOL, NULL, 0);
    keepbranch = cp_getvar("keep#branch", CP_BOOL, NULL, 0);

    /* Why bother printing out an empty plot? */
    if (!pl->pl_dvecs) {
        fprintf(cp_err, "Error: plot is empty, nothing written.\n");
        return;
    }

    if (raw_prec != -1) {
        prec = raw_prec;
    }
    else {
        prec = DEFPREC;
    }

    /* - Binary file binary write -  hvogt 15.03.2000 ---------------------*/
    if (binary) {
        if ((fp = fopen(name, app ? "ab" : "wb")) == NULL) {
            perror(name);
            return;
        }
        fprintf(cp_out, "binary raw file \"%s\"\n", name);
    }
    else {
        if ((fp = fopen(name, app ? "a" : "w")) == NULL) {
            perror(name);
            return;
        }
        fprintf(cp_out, "ASCII raw file \"%s\"\n", name);
    }
    /* --------------------------------------------------------------------*/

    numdims = nvars = length = 0;
    for (v = pl->pl_dvecs; v; v = v->v_next) {
        if (iscomplex(v)) {
            realflag = FALSE;
        }
        nvars++;
        /* Find the length and dimensions of the longest vector
         * in the plot.
         * Be paranoid and assume somewhere we may have
         * forgotten to set the dimensions of 1-D vectors.
         */
        if (v->v_numdims <= 1) {
            v->v_numdims = 1;
            v->v_dims[0] = v->v_length;
        }
        if (v->v_length > length) {
            length = v->v_length;
            numdims = v->v_numdims;
            for (j = 0; j < numdims; j++) {
                dims[j] = v->v_dims[j];
            }
        }
    }

    fprintf(fp, "Title: %s\n", pl->pl_title);
    fprintf(fp, "Date: %s\n", pl->pl_date);
    fprintf(fp, "Plotname: %s\n", pl->pl_name);
    fprintf(fp, "Flags: %s%s\n",
            realflag ? "real" : "complex", raw_padding ? "" : " unpadded");
    fprintf(fp, "No. Variables: %d\n", nvars);
    fprintf(fp, "No. Points: %d\n", length);
    if (numdims > 1) {
        dimstring(dims, numdims, buf);
        fprintf(fp, "Dimensions: %s\n", buf);
    }

    for (wl = pl->pl_commands; wl; wl = wl->wl_next) {
        fprintf(fp, "Command: %s\n", wl->wl_word);
    }

    for (vv = pl->pl_env; vv; vv = vv->va_next) {
        wl = cp_varwl(vv);
        if (vv->va_type == CP_BOOL) {
            fprintf(fp, "Option: %s\n", vv->va_name);
        }
        else {
            fprintf(fp, "Option: %s = ", vv->va_name);
            if (vv->va_type == CP_LIST) {
                fprintf(fp, "( ");
            }
            wl_print(wl, fp);
            if (vv->va_type == CP_LIST) {
                fprintf(fp, " )");
            }
            (void) putc('\n', fp);
        }
    }

    /* Before we write the stuff out, make sure that the scale is the first
     * in the list.
     */
    for (lv = NULL, v = pl->pl_dvecs; v != pl->pl_scale; v = v->v_next) {
        lv = v;
    }
    if (lv) {
        lv->v_next = v->v_next;
        v->v_next = pl->pl_dvecs;
        pl->pl_dvecs = v;
    }

    fprintf(fp, "Variables:\n");
    for (i = 0, v = pl->pl_dvecs; v; v = v->v_next) {
        /* write i(name) instead of name#branch */
        if (v->v_type == SV_CURRENT && !keepbranch) {
            branch = NULL;
            /* get name only*/
            if ((branch = strstr(v->v_name, "#branch")) != NULL) {
                *branch = '\0';
            }
            /* If the branch name is a number, the vector is already called i(name) */
            if (ciprefix("i(", v->v_name))
                fprintf(fp, "\t%d\t%s\t%s", i++, v->v_name, ft_typenames(v->v_type));
            else
                fprintf(fp, "\t%d\ti(%s)\t%s", i++, v->v_name, ft_typenames(v->v_type));
            /* restore name#branch */
            if (branch != NULL) *branch = '#';
        }
        /* write v(name)*/
        else if (v->v_type == SV_VOLTAGE) {
            /* If the node name is a number, the vector is already called v(name) */
            if (ciprefix("v(", v->v_name) || newcompat.eg) /* FIXME: newcompat.eg should be removed if EAGLE is updated */
               fprintf(fp, "\t%d\t%s\t%s", i++, v->v_name, ft_typenames(v->v_type));
            else
               fprintf(fp, "\t%d\tv(%s)\t%s", i++, v->v_name, ft_typenames(v->v_type));
        }
        /* write 'name' only for all other vector types */
        else {
            fprintf(fp, "\t%d\t%s\t%s", i++, v->v_name, ft_typenames(v->v_type));
        }
        if (v->v_flags & VF_MINGIVEN) {
            fprintf(fp, " min=%e", v->v_minsignal);
        }
        if (v->v_flags & VF_MAXGIVEN) {
            fprintf(fp, " max=%e", v->v_maxsignal);
        }
        if (v->v_defcolor) {
            fprintf(fp, " color=%s", v->v_defcolor);
        }
        if (v->v_gridtype) {
            fprintf(fp, " grid=%d", v->v_gridtype);
        }
        if (v->v_plottype) {
            fprintf(fp, " plot=%d", v->v_plottype);
        }
        /* Only write dims if they are different from default. */
        writedims = FALSE;
        if (v->v_numdims != numdims) {
            writedims = TRUE;
        }
        else {
            for (j = 0; j < numdims; j++)
                if (dims[j] != v->v_dims[j])
                    writedims = TRUE;
        }
        if (writedims) {
            dimstring(v->v_dims, v->v_numdims, buf);
            fprintf(fp, " dims=%s", buf);
        }
        (void) putc('\n', fp);
    }

    if (binary) {
        fprintf(fp, "Binary:\n");
        for (i = 0; i < length; i++) {
            for (v = pl->pl_dvecs; v; v = v->v_next) {
                /* Don't run off the end of this vector's data. */
                if (i < v->v_length) {
                    if (realflag) {
                        dd = (isreal(v) ? v->v_realdata[i] :
                              realpart(v->v_compdata[i]));
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                    }
                    else if (isreal(v)) {
                        dd = v->v_realdata[i];
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                        dd = 0.0;
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                    }
                    else {
                        dd = realpart(v->v_compdata[i]);
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                        dd = imagpart(v->v_compdata[i]);
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                    }
                }
                else if (raw_padding) {
                    dd = 0.0;
                    if (realflag) {
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                    }
                    else {
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                        (void) fwrite(&dd, sizeof(double), 1, fp);
                    }
                }
            }
        }
    }
    else {
        fprintf(fp, "Values:\n");
        for (i = 0; i < length; i++) {
            fprintf(fp, " %d", i);
            for (v = pl->pl_dvecs; v; v = v->v_next) {
                if (i < v->v_length) {
                    if (realflag) {
                        fprintf(fp, "\t%.*e\n", prec,
                                isreal(v) ? v->v_realdata[i] :
                                realpart(v->v_compdata[i]));
                    }
                    else if (isreal(v)) {
                        fprintf(fp, "\t%.*e,0.0\n", prec,
                                v->v_realdata[i]);
                    }
                    else {
                        fprintf(fp, "\t%.*e,%.*e\n", prec,
                                realpart(v->v_compdata[i]),
                                prec,
                                imagpart(v->v_compdata[i]));
                    }
                }
                else if (raw_padding) {
                    if (realflag) {
                        fprintf(fp, "\t%.*e\n", prec, 0.0);
                    }
                    else {
                        fprintf(fp, "\t%.*e,%.*e\n", prec, 0.0, prec, 0.0);
                    }
                }
            }
            (void) putc('\n', fp);
        }
    }
    (void) fclose(fp);
} /* end of function raw_write */



/* Read a raw file.  Returns a list of plot structures.  This routine should be
 * very flexible about what it expects to see in the rawfile.  Really all we
 * require is that there be one variables and one values section per plot
 * and that the variables precede the values.
 */

#define SKIP(s)                                 \
    skip_ws(skip_non_ws(s))

#define NONL(s)                                 \
    do {                                        \
        r = (s);                                \
        while (*r && (*r != '\n'))              \
            r++;                                \
        *r = '\0';                              \
    } while(0)

#define GETOUT()                                \
    do {                                        \
        fprintf(cp_err,                         \
                "Error: bad rawfile\n"          \
                "  point %d, var %s\n"          \
                "  load aborted\n",             \
                i, v->v_name);                  \
        return (NULL);                          \
    } while(0)


struct plot *
raw_read(char *name) {
    char *title = NULL;
    char *date = NULL;
    struct plot *plots = NULL, *curpl = NULL;
    char buf[BSIZE_SP], *s, *t, *r;
    int flags = 0, nvars = 0, npoints = 0, i, j;
    int ndimpoints, numdims = 0, dims[MAXDIMS];
    bool raw_padded = TRUE, is_ascii = FALSE;
    double junk;
    struct dvec *v, *nv;
    struct variable *vv;
    wordlist *wl, *nwl;
    FILE *fp, *lastin, *lastout, *lasterr;


    if ((fp = fopen(name, "rb")) == NULL) {
        perror(name);
        return NULL;
    }

    /* Since we call cp_evloop() from here, we have to do this junk. */
    lastin = cp_curin;
    lastout = cp_curout;
    lasterr = cp_curerr;
    cp_curin = cp_in;
    cp_curout = cp_out;
    cp_curerr = cp_err;

    cp_pushcontrol();

    while (fgets(buf, BSIZE_SP, fp)) {
        r = strchr(buf, '\n');
        if (r  &&  r > buf  &&  r[-1] == '\r') {
            r[-1] = '\n';
            r[0]  = '\0';
        }
        /* Figure out what this line is... */
        if (ciprefix("title:", buf)) {
            s = SKIP(buf);
            NONL(s);
            title = copy(s);
        } else if (ciprefix("date:", buf)) {
            s = SKIP(buf);
            NONL(s);
            date = copy(s);
        } else if (ciprefix("offset:", buf)) {
            fprintf(cp_err, "Warning: Offset: is not supported\n");
        } else if (ciprefix("plotname:", buf)) {
            s = SKIP(buf);
            NONL(s);
            if (curpl) {    /* reverse commands list */
                for (wl = curpl->pl_commands, curpl->pl_commands = NULL;
                     wl && wl->wl_next;
                     wl = nwl)
                {
                    nwl = wl->wl_next;
                    wl->wl_next = curpl->pl_commands;
                    curpl->pl_commands = wl;
                }
            }
            curpl = TMALLOC(struct plot, 1);
            curpl->pl_next = plots;
            plots = curpl;
            curpl->pl_name = copy(s);
            if (!date)
                date = copy(datestring());
            curpl->pl_date = date;
            if (!title)
                title = copy("default title");
            curpl->pl_title = title;
            curpl->pl_xdim2d = -1;
            curpl->pl_ydim2d = -1;
            date = NULL;
            title = NULL;
            flags = VF_PERMANENT;
            nvars = npoints = 0;
        } else if (ciprefix("flags:", buf)) {
            s = SKIP(buf);
            while ((t = gettok(&s)) != NULL) {
                if (cieq(t, "real"))
                    flags |= VF_REAL;
                else if (cieq(t, "complex"))
                    flags |= VF_COMPLEX;
                else if (cieq(t, "unpadded"))
                    raw_padded = FALSE;
                else if (cieq(t, "padded"))
                    raw_padded = TRUE;
                else
                    fprintf(cp_err, "Warning: unknown flag %s\n", t);
            }
        } else if (ciprefix("no. variables:", buf)) {
            s = SKIP(buf);
            s = SKIP(s);
            nvars = scannum(s);
        } else if (ciprefix("no. points:", buf)) {
            s = SKIP(buf);
            s = SKIP(s);
            npoints = scannum(s);
        } else if (ciprefix("dimensions:", buf)) {
            if (npoints == 0) {
                fprintf(cp_err,
                        "Error: misplaced Dimensions: line\n");
                continue;
            }
            s = SKIP(buf);
            if (atodims(s, dims, &numdims)) { /* Something's wrong. */
                fprintf(cp_err,
                        "Warning: syntax error in dimensions, ignored.\n");
                numdims = 0;
                continue;
            }
            if (numdims > MAXDIMS) {
                numdims = 0;
                continue;
            }
            /* Let's just make sure that the no. of points
             * and the dimensions are consistent.
             */
            for (j = 0, ndimpoints = 1; j < numdims; j++)
                ndimpoints *= dims[j];

            if (ndimpoints != npoints) {
                fprintf(cp_err,
                        "Warning: dimensions inconsistent with no. of points, ignored.\n");
                numdims = 0;
            }
        } else if (ciprefix("command:", buf)) {
            /* Note that we reverse these commands eventually... */
            s = SKIP(buf);
            NONL(s);
            if (curpl) {
                curpl->pl_commands = wl_cons(copy(s), curpl->pl_commands);
                wl = curpl->pl_commands;
            } else {
                fprintf(cp_err, "Error: misplaced Command: line\n");
            }
            /* Now execute the command if we can. */
            (void) cp_evloop(s);
        } else if (ciprefix("option:", buf)) {
            s = SKIP(buf);
            NONL(s);
            if (curpl) {
                wl = cp_lexer(s);
                for (vv = curpl->pl_env; vv && vv->va_next;
                     vv = vv->va_next)
                    ;
                if (vv)
                    vv->va_next = cp_setparse(wl);
                else
                    curpl->pl_env = cp_setparse(wl);
            } else {
                fprintf(cp_err, "Error: misplaced Option: line\n");
            }
        } else if (ciprefix("variables:", buf)) {
            /* We reverse the dvec list eventually... */
            if (!curpl) {
                fprintf(cp_err, "Error: no plot name given\n");
                plots = NULL;
                break;
            }
            s = SKIP(buf);
            if (!*s) {
                if (fgets(buf, BSIZE_SP, fp) == (char *) NULL) {
                    fprintf(cp_err, "Error: unable to read line\n");
                    plots = NULL;
                    break;
                }
                s = buf;
            }
            if (numdims == 0) {
                numdims = 1;
                dims[0] = npoints;
            }
            /* Now read all the variable lines in. */
            for (i = 0; i < nvars; i++) {
                /* Allocate the data array. We would use
                 * the desired vector length, but this would
                 * be dangerous if the file is invalid.
                 */
                v = dvec_alloc(NULL,
                               SV_NOTYPE, (short) flags,
                               npoints, NULL);
                /* Length and dims might be changed by options. */

                v->v_plot = curpl;

                if (!curpl->pl_scale)
                    curpl->pl_scale = v;

                if (!i) {
                    curpl->pl_scale = v;
                } else {
                    if (fgets(buf, BSIZE_SP, fp) == (char *) NULL) {
                        fprintf(cp_err, "Error: unable to read variable line\n");
                        break;
                    }
                    s = buf;
                }
                s = nexttok(s);  /* The strchr field. */
                if ((t = gettok(&s)) != NULL) {
                    v->v_name = t;
                } else {
                    fprintf(cp_err, "Error: bad var line %s\n", buf);
                    /* MW. v_name must be valid in the case that no. points = 0 */
                    v->v_name = copy("no vars\n");
                }
                t = gettok(&s); /* The type name. */
                if (t)
                    v->v_type = ft_typnum(t);
                else
                    fprintf(cp_err, "Error: bad var line %s\n", buf);

                /* Fix the name... */
                if (isdigit_c(v->v_name[0]) && (r = ft_typabbrev(v ->v_type)) != NULL) {
                    char *x = v->v_name;
                    v->v_name = tprintf("%s(%s)", r, v->v_name);
                    tfree(x);
                }

                /* Now come the strange options... */
                while ((t = gettok(&s)) != NULL) {
                    if (ciprefix("min=", t)) {
                        if (sscanf(t + 4, "%lf", &v->v_minsignal) != 1)
                            fprintf(cp_err, "Error: bad arg %s\n", t);
                        v->v_flags |= VF_MINGIVEN;
                    } else if (ciprefix("max=", t)) {
                        if (sscanf(t + 4, "%lf", &v->v_maxsignal) != 1)
                            fprintf(cp_err, "Error: bad arg %s\n", t);
                        v->v_flags |= VF_MAXGIVEN;
                    } else if (ciprefix("color=", t)) {
                        v->v_defcolor = copy(t + 6);
                    } else if (ciprefix("scale=", t)) {
                        // This cast is bad, but...
                        v->v_scale = (struct dvec *) copy(t + 6);
                    } else if (ciprefix("grid=", t)) {
                        v->v_gridtype = (GRIDTYPE) scannum(t + 5);
                    } else if (ciprefix("plot=", t)) {
                        v->v_plottype = (PLOTTYPE) scannum(t + 5);
                    } else if (ciprefix("dims=", t)) {
                        fixdims(v, t + 5);
                    } else {
                        fprintf(cp_err, "Warning: bad var param %s\n", t);
                    }
                }
                /* Now we default any missing dimensions. */
                if (!v->v_numdims) {
                    v->v_numdims = numdims;
                    for (j = 0; j < numdims; j++)
                        v->v_dims[j] = dims[j];
                }

                v->v_next = curpl->pl_dvecs;
                curpl->pl_dvecs = v;
            }

        } else if (ciprefix("values:", buf) || ciprefix("binary:", buf)) {

            if (!curpl) {
                fprintf(cp_err, "Error: no plot name given\n");
                plots = NULL;
                break;
            }

            /* We'd better reverse the dvec list now... */
            for (v = curpl->pl_dvecs, curpl->pl_dvecs = NULL; v; v = nv) {
                nv = v->v_next;
                v->v_next = curpl->pl_dvecs;
                curpl->pl_dvecs = v;
            }

            /* And fix the scale pointers. */
            for (v = curpl->pl_dvecs; v; v = v->v_next) {
                if (v->v_scale) {
                    for (nv = curpl->pl_dvecs; nv; nv = nv->v_next)
                        // This cast is bad, but...
                        if (cieq((char *) v->v_scale, nv->v_name)) {
                            v->v_scale = nv;
                            break;
                        }
                    if (!nv) {
                         // This cast is bad, but...
                        fprintf(cp_err,
                                "Error: no such vector %s\n", (char *) v->v_scale);
                        v->v_scale = NULL;
                    }
                }
            }

            if ((numdims == 2) && (flags & VF_REAL) &&
                    eq(curpl->pl_name, "Device Cross Section")) {
                if ((dims[0] > 1) && (dims[1] > 1) &&
                        (npoints == dims[0] * dims[1])) {
                    curpl->pl_xdim2d = dims[0];
                    curpl->pl_ydim2d = dims[1];
                }
            }
            if ((*buf == 'v') || (*buf == 'V'))
                is_ascii = TRUE;
            else
                is_ascii = FALSE;

            for (i = 0; i < npoints; i++) {
                if (is_ascii) {
                    /* It's an ASCII file. */
                    if (fscanf(fp, " %d", &j) != 1) {
                        fprintf(cp_err, "Error: unable to read point count\n");
                        break;
                    }
                    for (v = curpl->pl_dvecs; v; v = v->v_next) {
                        if (i < v->v_length) {
                            if (flags & VF_REAL) {
                                if (fscanf(fp, " %lf", &v->v_realdata[i]) != 1)
                                    GETOUT();
                            } else {
                                if (fscanf(fp, " %lf, %lf",
                                           &realpart(v->v_compdata[i]),
                                           &imagpart(v->v_compdata[i])) != 2)
                                    GETOUT();
                            }
                        } else if (raw_padded) {
                            if (flags & VF_REAL) {
                                if (fscanf(fp, " %lf", &junk) != 1)
                                    GETOUT();
                            } else {
                                if (fscanf(fp, " %lf, %lf", &junk, &junk) != 2)
                                    GETOUT();
                            }
                        }
                    }
                } else {
                    /* It's a Binary file. */
                    for (v = curpl->pl_dvecs; v; v = v->v_next) {
                        if (i < v->v_length) {
                            if (flags & VF_REAL) {
                                if (fread(&v->v_realdata[i],
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                            } else {
                                if (fread(&v->v_compdata[i].cx_real,
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                                if (fread(&v->v_compdata[i].cx_imag,
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                            }
                        } else if (raw_padded) {
                            if (flags & VF_REAL) {
                                if (fread(&junk,
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                            } else {
                                if (fread(&junk,
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                                if (fread(&junk,
                                          sizeof(double), 1, fp) != 1)
                                    GETOUT();
                            }
                        }
                    }
                }
            }
        } else {
            s = buf;
            if (is_ascii) {
                s = SKIP(s);
            }
            if (*s) {
                fprintf(cp_err,
                        "Error: strange line in rawfile:\n  %s\n  load aborted.\n", buf);
                return (NULL);
            }
        }
    } /* end of loop */

    if (curpl) {    /* reverse commands list */
        for (wl = curpl->pl_commands, curpl->pl_commands = NULL;
             wl && wl->wl_next;
             wl = nwl)
        {
            nwl = wl->wl_next;
            wl->wl_next = curpl->pl_commands;
            curpl->pl_commands = wl;
        }
    }

    /* Fix everything up nicely again. */
    cp_popcontrol();
    cp_curin = lastin;
    cp_curout = lastout;
    cp_curerr = lasterr;
    (void) fclose(fp);
    return (plots);
}


/* s is a string of the form d1,d2,d3... */

static void
fixdims(struct dvec *v, char *s)
{
    int i, ndimpoints;

    if (atodims(s, v->v_dims, &(v->v_numdims))) { /* Something's wrong. */
        fprintf(cp_err,
                "Warning: syntax error in dimensions, ignored.\n");
        return;
    } else if (v->v_numdims > MAXDIMS) {
        return;
    }

    /* If the no. of points is less than the the total data length,
     * truncate the vector length.  If it's greater in length, we
     * have serious problems with this vector.  Try to fix
     * by setting to default dimensions when we return.
     */
    for (i = 0, ndimpoints = 1; i < v->v_numdims; i++)
        ndimpoints *= v->v_dims[i];

    if (v->v_length >= ndimpoints)
        dvec_trunc(v, ndimpoints);
    else
        v->v_numdims = 0;
}


/* Write a s-param file according to format Touchstone Vers. 1. */

void
spar_write(char *name, struct plot *pl, double Rbaseval)
{
    FILE *fp;
    int length;
    int i, prec;
    struct dvec *v, *lv;

    /* Why bother printing out an empty plot? */
    if (!pl->pl_dvecs) {
        fprintf(cp_err, "Error writing s2p: plot is empty, nothing written.\n");
        return;
    }

    if (raw_prec != -1)
        prec = raw_prec;
    else
        prec = 6;

    length = 0;
    for (v = pl->pl_dvecs; v; v = v->v_next) {

        /* All vectors have to have same length,
           only dimension 1 is allowed */
        if (length == 0)
            length = v->v_length;

        if (length != v->v_length) {
            fprintf(stderr, "Error writing s2p: lentgth of vector %s differs from length of vector 'frequency'\n",
                    v->v_name);
            return;
        }
        if (v->v_numdims != 1) {
            fprintf(stderr, "Error writing s2p: Dimension of vector %s greater than 1\n", v->v_name);
            return;
        }

        /* Find the length and dimensions of the longest vector
         * in the plot.
         * Be paranoid and assume somewhere we may have
         * forgotten to set the dimensions of 1-D vectors.

         if (v->v_numdims <= 1) {
         v->v_numdims = 1;
         v->v_dims[0] = v->v_length;
         }
         if (v->v_length > length) {
         length = v->v_length;
         numdims = v->v_numdims;
         for (j = 0; j < numdims; j++) {
         dims[j] = v->v_dims[j];
         }
         }*/
    }

    if ((fp = fopen(name, "w")) == NULL) {
        perror(name);
        return;
    }

    fprintf(fp, "!2-port S-parameter file\n");
    fprintf(fp, "!Title: %s\n", pl->pl_title);
    fprintf(fp, "!Generated by ngspice at %s\n", pl->pl_date);
    fprintf(fp, "# Hz S RI R %g\n", Rbaseval);
    fprintf(fp, "!%-*.5s  %-*.5s  %-*.5s  %-*.5s  %-*.5s  %-*.5s  %-*.5s  %-*.5s  %-*.5s\n",
            prec+8, "freq",
            prec+8, "ReS11", prec+8, "ImS11", prec+8, "ReS21", prec+8, "ImS21",
            prec+8, "ReS12", prec+8, "ImS12", prec+8, "ReS22", prec+8, "ImS22");

    /* Before we write the stuff out, make sure that the scale is the first
     * in the list.
     */
    for (lv = NULL, v = pl->pl_dvecs; v != pl->pl_scale; v = v->v_next)
        lv = v;
    if (lv) {
        lv->v_next = v->v_next;
        v->v_next = pl->pl_dvecs;
        pl->pl_dvecs = v;
    }

    /* print frequency first as real value, the real and imag part of
       S11, S21, S12, S22 respectively */
    for (i = 0; i < length; i++) {
        for (v = pl->pl_dvecs; v; v = v->v_next) {
            if (i < v->v_length) {
                if (cieq(v->v_name, "frequency"))
                    fprintf(fp, "% .*e  ", prec,
                            realpart(v->v_compdata[i]));
                else
                    fprintf(fp, "% .*e  % .*e  ", prec,
                            realpart(v->v_compdata[i]),
                            prec,
                            imagpart(v->v_compdata[i]));
            }
        }
        (void) putc('\n', fp);
    }

    (void) fclose(fp);
}
