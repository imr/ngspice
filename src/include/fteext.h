/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 1999 Paolo Nenzi - 2000 AlansFixes
**********/

/*
 * Definitions for all external symbols in FTE.
 */

#ifndef FTEext_h
#define FTEext_h

#include <config.h>

/* needed to find out what the interface structures look like */
#include "typedefs.h"
#include "ifsim.h"
#include "dvec.h"
#include "plot.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "fteinp.h"

/* arg.c */

extern void arg_plot(wordlist *wl, struct comm *command);
extern void arg_display(wordlist *wl, struct comm *command);
extern void arg_print(wordlist *wl, struct comm *command);
extern void arg_let(wordlist *wl, struct comm *command);
extern void arg_load(wordlist *wl, struct comm *command);
extern void arg_set(wordlist *wl, struct comm *command);
extern void outmenuprompt(char *string);

/* aspice.c */

extern void com_aspice(wordlist *wl);
extern void com_jobs(wordlist *wl);
extern void com_rspice(wordlist *wl);
extern void ft_checkkids(void);

/* breakpoint.c */

extern bool ft_bpcheck(struct plot *runplot, int iteration);
extern void com_delete(wordlist *wl);
extern void com_iplot(wordlist *wl);
extern void com_save(wordlist *wl);
extern void com_save2(wordlist *wl, char *name);
extern void com_step(wordlist *wl);
extern void com_stop(wordlist *wl);
extern void com_sttus(wordlist *wl);
extern void com_trce(wordlist *wl);
extern void ft_trquery(void);
extern void dbfree(struct dbcomm *db);


/* breakp2.c */

extern int ft_getSaves(struct save_info **);


/* circuits.c */

extern struct circ *ft_curckt;
extern struct circ *ft_circuits;
extern struct subcirc *ft_subcircuits;
extern void ft_newcirc(struct circ *ckt);

/* clip.c */

extern bool clip_line(int *pX1, int *pY1, int *pX2, int *pY2, int l, int b, int r, int t);
extern bool clip_to_circle(int *x1, int *y1, int *x2, int *y2, int cx, int cy, int rad);

/* cmath1.c */

extern bool cx_degrees;
extern void *cx_mag(void *, short int , int , int *, short int *);
extern void *cx_ph(void *, short int , int , int *, short int *);
extern void *cx_j(void *, short int , int , int *, short int *);
extern void *cx_real(void *, short int , int , int *, short int *);
extern void *cx_imag(void *, short int , int , int *, short int *);
extern void *cx_pos(void *, short int , int , int *, short int *);
extern void *cx_db(void *, short int , int , int *, short int *);
extern void *cx_log(void *, short int , int , int *, short int *);
extern void *cx_ln(void *, short int , int , int *, short int *);
extern void *cx_exp(void *, short int , int , int *, short int *);
extern void *cx_sqrt(void *, short int , int , int *, short int *);
extern void *cx_sin(void *, short int , int , int *, short int *);
extern void *cx_cos(void *, short int , int , int *, short int *);

/* cmath2.c */

extern void *cx_tan(void *, short int , int , int *, short int *);
extern void *cx_atan(void *, short int , int , int *, short int *);
extern void *cx_norm(void *, short int , int , int *, short int *);
extern void *cx_uminus(void *, short int , int , int *, short int *);
extern void *cx_rnd(void *, short int , int , int *, short int *);
extern void *cx_sunif(void *, short int , int , int *, short int *);
extern void *cx_sgauss(void *, short int , int , int *, short int *);
extern void *cx_mean(void *, short int , int , int *, short int *);
extern void *cx_avg(void *, short int , int , int *, short int *);
extern void *cx_length(void *, short int , int , int *, short int *);
extern void *cx_vector(void *, short int , int , int *, short int *);
extern void *cx_unitvec(void *, short int , int , int *, short int *);
 
/* Routoure JM : somme useful functions */
extern void *cx_min(void *, short int , int , int *, short int *);
extern void *cx_max(void *, short int , int , int *, short int *);
extern void *cx_d(void *, short int , int , int *, short int *);

extern void *cx_plus(void *, void *, short int , short int , int);
extern void *cx_minus(void *, void *, short int , short int , int);
extern void *cx_times(void *, void *, short int , short int , int);
extern void *cx_mod(void *, void *, short int , short int , int);

/* cmath3.c */

extern void *cx_divide(void *, void *, short int , short int , int);
extern void *cx_comma(void *, void *, short int , short int , int);
extern void *cx_power(void *, void *, short int , short int , int);
extern void *cx_eq(void *, void *, short int , short int , int);
extern void *cx_gt(void *, void *, short int , short int , int);
extern void *cx_lt(void *, void *, short int , short int , int);
extern void *cx_ge(void *, void *, short int , short int , int);
extern void *cx_le(void *, void *, short int , short int , int);
extern void *cx_ne(void *, void *, short int , short int , int);

/* cmath4.c */

extern void *cx_and(void *, void *, short int , short int , int);
extern void *cx_or(void *, void *, short int , short int , int);
extern void *cx_not(void *, short int , int , int *, short int *);

extern void *cx_interpolate(void *, short int , int , int *, short int *, struct plot *, struct plot *, int );
extern void *cx_deriv(void *, short int , int , int *, short int *, struct plot *, struct plot *, int );
extern void *cx_group_delay(void *, short int , int , int *, short int *, struct plot *, struct plot *, int );


/* cmdtab.c */

extern struct comm *cp_coms;

/* compose.c */

extern void com_compose(wordlist *wl);

/* cpinterface.c symbols declared in CPextern.h */

/* debugcoms.c */

extern void com_dump(wordlist *wl);
extern void com_state(wordlist *wl);

/* define.c */

extern struct pnode *ft_substdef(const char *name, struct pnode *args);
extern void com_define(wordlist *wl);
extern void com_undefine(wordlist *wl);
extern void ft_pnode(struct pnode *pn);

/* device.c */

extern void com_show(wordlist *wl);
extern void com_showmod(wordlist *wl);
extern void com_alter(wordlist *wl);
extern void com_altermod(wordlist *wl);

/* diff.c */

extern void com_diff(wordlist *wl);

/* doplot.c */

extern void com_asciiplot(wordlist *wl);
extern void com_hardcopy(wordlist *wl);
extern void com_plot(wordlist *wl);
extern void com_xgraph(wordlist *wl);

/* dotcards.c */

extern bool ft_acctprint;
extern bool ft_noacctprint;
extern bool ft_noinitprint;
extern bool ft_listprint;
extern bool ft_nopage;
extern bool ft_nomod;
extern bool ft_nodesprint;
extern bool ft_optsprint;
extern int ft_cktcoms(bool terse);
extern void ft_dotsaves(void);
extern int ft_savedotargs(void);

/* error.c */

extern void fatal(void);
extern void fperror(char *mess, int code);
extern void ft_sperror(int code, char *mess);
extern char ErrorMessage[];
extern void internalerror(char *); 
extern void externalerror(char *); 



/* evaluate.c */

extern struct dvec *op_and(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_comma(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_divide(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_eq(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *ft_evaluate(struct pnode *node);
extern struct dvec *op_ge(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_gt(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_le(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_lt(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_minus(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_mod(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_ne(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_not(struct pnode *arg);
extern struct dvec *op_or(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_ind(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_plus(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_power(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_times(struct pnode *arg1, struct pnode *arg2);
extern struct dvec *op_uminus(struct pnode *arg);
extern struct dvec *op_range(struct pnode *arg1, struct pnode *arg2);

/* spec.c */
extern void com_spec(wordlist *wl);

/* com_fft.c */
extern void com_fft(wordlist*);

/* com_sysinfo.c */
extern void com_sysinfo(wordlist *wl);

/* ginterface.c 

extern bool gi_init();
extern bool gi_endpause;
extern bool gi_rottext;
extern int gi_fntheight;
extern int gi_fntwidth;
extern int gi_maxx;
extern int gi_maxy;
extern int gi_nolst;
extern int gi_nocolors;
extern int gi_package;
extern void gi_arc();
extern void gi_clearscreen();
extern void gi_close();
extern void gi_drawline();
extern void gi_redraw();
extern void gi_setcolor();
extern void gi_resetcolor();
extern void gi_setlinestyle();
extern void gi_text();
extern void gi_update();
*/

/* graf.c */

extern bool gr_gmode;
extern bool gr_hmode;
extern void gr_clean(void);
extern void gr_end(struct dvec *dv);
extern void gr_iplot(struct plot *plot);
extern void gr_pmsg(char *text);
extern void gr_point(struct dvec *dv, double newx, double newy, double oldx, double oldy, int np);
extern void gr_start(struct dvec *dv);
extern double gr_xrange[2];
extern double gr_yrange[2];
extern int gr_xmargin;
extern int gr_ymargin;
extern int gr_xcenter;
extern int gr_ycenter;
extern int gr_radius;
extern bool gr_circular;

/* grid.c */

extern void gr_fixgrid(GRAPH *graph, double xdelta, double ydelta, int xtype, int ytype);

/* inp.c */

extern void com_edit(wordlist *wl);
extern void com_listing(wordlist *wl);
extern void com_source(wordlist *wl);
void inp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		struct line *options, char *filename);
extern void inp_source(char *file);
void inp_spsource(FILE *fp, bool comfile, char *filename);
extern void inp_casefix(char *string);
extern void inp_list(FILE *file, struct line *deck, struct line *extras, int type);
extern void inp_readall(FILE *fp, struct line **data, int call_depth, char *dir_name, bool comfile);
extern FILE *inp_pathopen(char *name, char *mode);

/* nutinp.c */

void inp_nutsource(FILE *fp, bool comfile, char *filename);
void nutinp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		   struct line *options, char *filename);
extern void nutcom_source(wordlist *wl);

/* interpolate.c */

extern bool ft_interpolate(double *data, double *ndata, double *oscale, int olen, double *nscale, int nlen, int degree);
extern bool ft_polyfit(double *xdata, double *ydata, double *result, int degree, double *scratch);
extern double ft_peval(double x, double *coeffs, int degree);
extern void ft_polyderiv(double *coeffs, int degree);
extern void com_linearize(wordlist *wl);

/* misccoms.c */

extern void com_bug(wordlist *wl);
extern void com_ahelp(wordlist *wl);
extern void com_ghelp(wordlist *wl);
extern void com_help(wordlist *wl);
extern void com_quit(wordlist *wl);
extern void com_version(wordlist *wl);
extern int  hcomp(const void *a, const void *b);
extern void com_where(wordlist *wl);

/* mw_coms.c */
extern void com_removecirc(wordlist *wl);

/* numparse.c */

extern bool ft_strictnumparse;
double * ft_numparse(char **s, bool whole);

/* options.c */

extern bool ft_simdb;
extern bool ft_parsedb;
extern bool ft_evdb;
extern bool ft_vecdb;
extern bool ft_grdb;
extern bool ft_gidb;
extern bool ft_controldb;
extern bool ft_asyncdb;
extern char *ft_setkwords[];
extern struct line *inp_getopts(struct line *deck);
extern struct line *inp_getoptsc(char *in_line, struct line *com_options);
extern struct variable *cp_enqvar(char *word);
extern bool ft_ngdebug;

/* parse.c */

extern struct func ft_funcs[];
extern struct func func_not;
extern struct func func_uminus;
extern struct pnode * ft_getpnames(wordlist *wl, bool check);
#define free_pnode(ptr)  free_pnode_x(ptr); ptr=NULL
extern void free_pnode_x(struct pnode *t);

/* plotcurve.c */

extern int ft_findpoint(double pt, double *lims, int maxp, int minp, bool islog);
extern double * ft_minmax(struct dvec *v, bool real);
extern void ft_graf(struct dvec *v, struct dvec *xs, bool nostart);

/* postcoms.c */

extern void com_cross(wordlist *wl);
extern void com_display(wordlist *wl);
extern void com_let(wordlist *wl);
extern void com_unlet(wordlist *wl);
extern void com_load(wordlist *wl);
extern void com_print(wordlist *wl);
extern void com_write(wordlist *wl);
extern void com_destroy(wordlist *wl);
extern void com_splot(wordlist *wl);
extern void com_setscale(wordlist *wl);
extern void com_transpose(wordlist *wl);

/* rawfile.c */
extern int raw_prec;
extern void raw_write(char *name, struct plot *pl, bool app, bool binary);
extern struct plot *raw_read(char *name);

/* meas.c */
extern void do_measure(char *what, bool chk_only);
extern bool check_autostop(char *what);
extern void com_meas(wordlist *wl);

/* randnumb.c */
extern void TausSeed(void);
/* resource.c */

extern void com_rusage(wordlist *wl);
extern void ft_ckspace(void);
extern void init_rlimits(void);

/* runcoms.c */

extern void com_ac(wordlist *wl);
extern void com_dc(wordlist *wl);
extern void com_op(wordlist *wl);
extern void com_pz(wordlist *wl);
extern void com_sens(wordlist *wl);
extern void com_rset(wordlist *wl);
extern void com_resume(wordlist *wl);
extern void com_run(wordlist *wl);
extern void com_tran(wordlist *wl);
extern void com_tf(wordlist *wl);
extern void com_scirc(wordlist *wl);
extern void com_disto(wordlist *wl);
extern void com_noise(wordlist *wl);
extern int ft_dorun(char *file);

extern bool ft_getOutReq(FILE **, struct plot **, bool *, char *, char *);

/* spice.c & nutmeg.c */

extern bool ft_nutmeg;
extern IFsimulator *ft_sim;
extern char *ft_rawfile;
extern char *cp_program;
extern RETSIGTYPE ft_sigintr(void);
extern RETSIGTYPE sigfloat(int sig, int code);
extern RETSIGTYPE sigstop(void);
extern RETSIGTYPE sigill(void);
extern RETSIGTYPE sigbus(void);
extern RETSIGTYPE sigsegv(void);
extern RETSIGTYPE sig_sys(void);
extern int main(int argc, char **argv);

/* spiceif.c & nutmegif.c */

extern bool if_tranparams(struct circ *ci, double *start, double *stop, double *step);
extern char *if_errstring(int code);
extern CKTcircuit *if_inpdeck(struct line *deck, INPtables **tab);
extern int if_run(CKTcircuit *t, char *what, wordlist *args, INPtables *tab);
extern int if_sens_run(CKTcircuit *t, wordlist *args, INPtables *tab);
extern struct variable *(*if_getparam)(CKTcircuit *ckt, char** name, char* param, int ind, int do_model);
extern struct variable * nutif_getparam(CKTcircuit *ckt, char **name, char *param, int ind, int do_model);
extern struct variable *spif_getparam(CKTcircuit *ckt, char **name, char *param, int ind, int do_model);
extern struct variable *spif_getparam_special(CKTcircuit *ckt, char **name, char *param, int ind, int do_model);
extern void if_cktfree(CKTcircuit *ckt, INPtables *tab);
extern void if_dump(CKTcircuit *ckt, FILE *file);
extern int if_option(CKTcircuit *ckt, char *name, enum cp_types type, void *value);
extern void if_setndnames(char *line);
extern void if_setparam_model(CKTcircuit *ckt, char **name, char *val );
extern void if_setparam(CKTcircuit *ckt, char **name, char *param, struct dvec *val, int do_model);
extern struct variable *if_getstat(CKTcircuit *ckt, char *name);

/* subckt.c */

extern struct line *inp_deckcopy(struct line *deck);
extern struct line *inp_subcktexpand(struct line *deck);

/* typesdef.c */

extern void com_dftype(wordlist *);
extern void com_stype(wordlist *);
extern char *ft_typabbrev(int);
extern char *ft_typenames(int);
extern char *ft_plotabbrev(char *);
extern int ft_typnum(char *);

/* vectors.c */

extern bool vec_eq(struct dvec *v1, struct dvec *v2);
extern int plot_num;
extern struct dvec *vec_fromplot(char *word, struct plot *plot);
extern struct dvec *vec_copy(struct dvec *v);
extern struct dvec *vec_get(const char *word);
extern struct dvec *vec_mkfamily(struct dvec *v);
extern struct plot *plot_cur;
extern struct plot *plot_alloc(char *name);
extern struct plot *plot_list;
extern int plotl_changed;
extern void plot_add(struct plot *pl);
#define vec_free(ptr)  vec_free_x(ptr); ptr=NULL
extern void vec_free_x(struct dvec *v);
extern void vec_gc(void);
extern void ft_loadfile(char *file);
extern void vec_new(struct dvec *d);
extern void plot_docoms(wordlist *wl);
extern void vec_remove(char *name);
extern void plot_setcur(char *name);
extern void plot_new(struct plot *pl);
extern char *vec_basename(struct dvec *v);
extern bool plot_prefix(char *pre, char *str);
extern void vec_transpose(struct dvec *v);

/* main.c */
extern bool ft_intrpt;
extern bool ft_setflag;

/* newcoms.c */
extern void com_reshape(wordlist *wl);

/* dimens.c */
extern void dimstring(int *data, int length, char *retstring);
extern int atodims(char *p, int *data, int *outlength);
extern void indexstring(int *data, int length, char *retstring);
extern int incindex(int *counts, int numcounts, int *dims, int numdims);

#endif /* FTEext_h */
