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
#include "ifsim.h"
#include "dvec.h"
#include "plot.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "fteinp.h"

/* arg.c */

extern void arg_plot();
extern void arg_display();
extern void arg_print();
extern void arg_let();
extern void arg_load();
extern void arg_set();
extern void outmenuprompt();

/* aspice.c */

extern void com_aspice();
extern void com_jobs();
extern void com_rspice();
extern void ft_checkkids();

/* binary.c */

extern void braw_write();
extern struct plot *braw_read();

/* breakpoint.c */

extern bool ft_bpcheck();
extern void com_delete();
extern void com_iplot();
extern void com_save();
extern void com_save2(wordlist *, char *);
extern void com_step();
extern void com_stop();
extern void com_sttus();
extern void com_trce();
extern void ft_trquery();
extern void dbfree( );


/* breakp2.c */

extern int ft_getSaves(struct save_info **);


/* circuits.c */

extern struct circ *ft_curckt;
extern struct circ *ft_circuits;
extern struct subcirc *ft_subcircuits;
extern void ft_newcirc();

/* clip.c */

extern bool clip_line();
extern bool clip_to_circle();

/* cmath1.c */

extern bool cx_degrees;
extern void *cx_mag(void *, short int , int , int *, short int *, ...);
extern void *cx_ph(void *, short int , int , int *, short int *, ...);
extern void *cx_j(void *, short int , int , int *, short int *, ...);
extern void *cx_real(void *, short int , int , int *, short int *, ...);
extern void *cx_imag(void *, short int , int , int *, short int *, ...);
extern void *cx_pos(void *, short int , int , int *, short int *, ...);
extern void *cx_db(void *, short int , int , int *, short int *, ...);
extern void *cx_log(void *, short int , int , int *, short int *, ...);
extern void *cx_ln(void *, short int , int , int *, short int *, ...);
extern void *cx_exp(void *, short int , int , int *, short int *, ...);
extern void *cx_sqrt(void *, short int , int , int *, short int *, ...);
extern void *cx_sin(void *, short int , int , int *, short int *, ...);
extern void *cx_cos(void *, short int , int , int *, short int *, ...);

/* cmath2.c */

extern void *cx_tan(void *, short int , int , int *, short int *, ...);
extern void *cx_atan(void *, short int , int , int *, short int *, ...);
extern void *cx_norm(void *, short int , int , int *, short int *, ...);
extern void *cx_uminus(void *, short int , int , int *, short int *, ...);
extern void *cx_rnd(void *, short int , int , int *, short int *, ...);
extern void *cx_mean(void *, short int , int , int *, short int *, ...);
extern void *cx_length(void *, short int , int , int *, short int *, ...);
extern void *cx_vector(void *, short int , int , int *, short int *, ...);
extern void *cx_unitvec(void *, short int , int , int *, short int *, ...);
 
/* Routoure JM : somme useful functions */
extern void *cx_min(void *, short int , int , int *, short int *, ...);
extern void *cx_max(void *, short int , int , int *, short int *, ...);
extern void *cx_d(void *, short int , int , int *, short int *, ...);

extern void *cx_plus(void *, void *, short int , short int , int, ...);
extern void *cx_minus(void *, void *, short int , short int , int, ...);
extern void *cx_times(void *, void *, short int , short int , int, ...);
extern void *cx_mod(void *, void *, short int , short int , int, ...);

/* cmath3.c */

extern void *cx_divide(void *, void *, short int , short int , int, ...);
extern void *cx_comma(void *, void *, short int , short int , int, ...);
extern void *cx_power(void *, void *, short int , short int , int, ...);
extern void *cx_eq(void *, void *, short int , short int , int, ...);
extern void *cx_gt(void *, void *, short int , short int , int, ...);
extern void *cx_lt(void *, void *, short int , short int , int, ...);
extern void *cx_ge(void *, void *, short int , short int , int, ...);
extern void *cx_le(void *, void *, short int , short int , int, ...);
extern void *cx_ne(void *, void *, short int , short int , int, ...);

/* cmath4.c */

extern void *cx_and(void *, void *, short int , short int , int, ...);
extern void *cx_or(void *, void *, short int , short int , int, ...);
extern void *cx_not(void *, short int , int , int *, short int * , ...);
extern void *cx_interpolate(void *, short int , int , int *, short int *, ...); /* struct plot *, struct plot *, int ); */
extern void *cx_deriv(void *, short int , int , int *, short int *, ...); /*struct plot *, struct plot *, int );*/

/* cmdtab.c */

extern struct comm *cp_coms;

/* compose.c */

extern void com_compose();

/* cpinterface.c symbols declared in CPextern.h */

/* debugcoms.c */

extern void com_dump();
extern void com_state();

/* define.c */

extern struct pnode *ft_substdef();
extern void com_define();
extern void com_undefine();
extern void ft_pnode();

/* device.c */

extern void com_show();
extern void com_showmod();
extern void com_alter();
extern void com_altermod();

/* diff.c */

extern void com_diff();

/* doplot.c */

extern void com_asciiplot();
extern void com_hardcopy();
extern void com_plot();
extern void com_xgraph();

/* dotcards.c */

extern bool ft_acctprint;
extern bool ft_listprint;
extern bool ft_nopage;
extern bool ft_nomod;
extern bool ft_nodesprint;
extern bool ft_optsprint;
extern int ft_cktcoms(bool terse);
extern void ft_dotsaves();
extern int ft_savedotargs();

/* error.c */

extern void fatal();
extern void fperror();
extern void ft_sperror();
extern char ErrorMessage[];
extern void internalerror(char *); 
extern void externalerror(char *); 



/* evaluate.c */

extern struct dvec *op_and();
extern struct dvec *op_comma();
extern struct dvec *op_divide();
extern struct dvec *op_eq();
extern struct dvec *ft_evaluate();
extern struct dvec *op_ge();
extern struct dvec *op_gt();
extern struct dvec *op_le();
extern struct dvec *op_lt();
extern struct dvec *op_minus();
extern struct dvec *op_mod();
extern struct dvec *op_ne();
extern struct dvec *op_not();
extern struct dvec *op_or();
extern struct dvec *op_ind();
extern struct dvec *op_plus();
extern struct dvec *op_power();
extern struct dvec *op_times();
extern struct dvec *op_uminus();
extern struct dvec *op_range();


/* spec.c */

extern void com_spec();

/* ginterface.c */

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

/* graf.c */

extern bool gr_gmode;
extern bool gr_hmode;
extern void gr_clean();
extern void gr_end();
extern void gr_iplot();
extern void gr_iplot_end();
extern void gr_pmsg();
extern void gr_point();
extern void gr_start();
extern double gr_xrange[2];
extern double gr_yrange[2];
extern int gr_xmargin;
extern int gr_ymargin;
extern int gr_xcenter;
extern int gr_ycenter;
extern int gr_radius;
extern bool gr_circular;

/* grid.c */

extern void gr_fixgrid();

/* inp.c */

extern void com_edit();
extern void com_listing();
extern void com_source();
void inp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		struct line *options, char *filename);
extern void inp_source();
void inp_spsource(FILE *fp, bool comfile, char *filename);
extern void inp_casefix();
extern void inp_list();
extern void inp_readall();
extern FILE *inp_pathopen();

/* nutinp.c */

void inp_nutsource(FILE *fp, bool comfile, char *filename);
void nutinp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		   struct line *options, char *filename);
extern void nutcom_source();

/* interpolate.c */

extern bool ft_interpolate();
extern bool ft_polyfit();
extern double ft_peval();
extern void ft_polyderiv( );
extern void com_linearize();

/* mfbinterface.c */

extern void mi_arc();
extern bool mi_init();
extern void mi_clearscreen();
extern void mi_close();
extern void mi_drawline();
extern void mi_resetcolor();
extern void mi_setcolor();
extern void mi_setlinestyle();
extern void mi_text();
extern void mi_update();

/* misccoms.c */

extern void com_bug();
extern void com_ahelp();
extern void com_ghelp();
extern void com_help();
extern void com_quit();
extern void com_version();
extern int  hcomp();
extern void com_where();

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
extern struct line *inp_getopts();
extern struct variable *cp_enqvar();
extern struct variable *cp_uservars();
extern int cp_userset();

/* parse.c */

extern struct func ft_funcs[];
extern struct func func_not;
extern struct func func_uminus;
extern struct pnode * ft_getpnames(wordlist *wl, bool check);
#define free_pnode(ptr)  free_pnode_x(ptr); ptr=NULL
extern void free_pnode_x();

/* plotcurve.c */

extern int ft_findpoint(double pt, double *lims, int maxp, int minp, bool islog);
extern double * ft_minmax(struct dvec *v, bool real);
extern void ft_graf(struct dvec *v, struct dvec *xs, bool nostart);

/* plotinterface.c */

extern void pi_arc();
extern bool pi_init();
extern void pi_clearscreen();
extern void pi_close();
extern void pi_drawline();
extern void pi_resetcolor();
extern void pi_setcolor();
extern void pi_setlinestyle();
extern void pi_text();
extern void pi_update();

/* postcoms.c */

extern void com_cross();
extern void com_display();
extern void com_let();
extern void com_unlet();
extern void com_load();
extern void com_print();
extern void com_write();
extern void com_destroy();
extern void com_splot();
extern void com_setscale();
extern void com_transpose();

/* rawfile.c */
extern int raw_prec;
extern void raw_write(char *name, struct plot *pl, bool app, bool binary);
extern struct plot *raw_read();

/* resource.c */

extern void com_rusage();
extern void ft_ckspace();
extern void init_rlimits();

/* runcoms.c */

extern void com_ac();
extern void com_dc();
extern void com_op();
extern void com_pz();
extern void com_sens();
extern void com_rset();
extern void com_resume();
extern void com_run();
extern void com_tran();
extern void com_tf();
extern void com_scirc();
extern void com_disto();
extern void com_noise();
extern int ft_dorun();

extern bool ft_getOutReq(FILE **, struct plot **, bool *, char *, char *);

/* spice.c & nutmeg.c */

extern bool ft_nutmeg;
extern IFsimulator *ft_sim;
extern char *ft_rawfile;
extern char *cp_program;
extern RETSIGTYPE ft_sigintr();
extern RETSIGTYPE sigfloat();
extern RETSIGTYPE sigstop();
extern RETSIGTYPE sigquit();
extern RETSIGTYPE sigill();
extern RETSIGTYPE sigbus();
extern RETSIGTYPE sigsegv();
extern RETSIGTYPE sig_sys();
extern int main();

/* spiceif.c & nutmegif.c */

extern bool if_tranparams();
extern char *if_errstring();
extern char *if_inpdeck();
extern int if_run();
extern int if_sens_run();
extern struct variable *(*if_getparam)();
extern struct variable *nutif_getparam();
extern struct variable *spif_getparam();
extern void if_cktfree();
extern void if_dump();
extern int if_option();
extern void if_setndnames();
extern void if_setparam();
extern struct variable *if_getstat();

/* subckt.c */

extern struct line *inp_deckcopy();
extern struct line *inp_subcktexpand();

/* types.c */

extern void com_dftype();
extern void com_stype();
extern char *ft_typabbrev();
extern char *ft_typenames();
extern char *ft_plotabbrev();
extern int ft_typnum();

/* vectors.c */

extern bool vec_eq();
extern int plot_num;
extern struct dvec *vec_fromplot();
extern struct dvec *vec_copy();
extern struct dvec *vec_get();
extern struct dvec *vec_mkfamily();
extern struct plot *plot_cur;
extern struct plot *plot_alloc();
extern struct plot *plot_list;
extern int plotl_changed;
extern void plot_add();
#define vec_free(ptr)  vec_free_x(ptr); ptr=NULL
extern void vec_free_x();
extern void vec_gc();
extern void ft_loadfile();
extern void vec_new();
extern void plot_docoms();
extern void vec_remove();
extern void ft_sdatafree();
extern void plot_setcur();
extern void plot_new();
extern char *vec_basename();
extern bool plot_prefix();
extern void vec_transpose();

/* writedata.c */

extern bool ft_intrpt;
extern bool ft_setflag;
extern int wrd_close();
extern int wrd_command();
extern int wrd_cptime;
extern int wrd_end();
extern int wrd_init();
extern int wrd_limpts;
extern int wrd_open();
extern int wrd_output();
extern int wrd_point();
extern int wrd_pt2();
extern int wrd_run();
extern int wrd_stopnow();
extern void wrd_chtrace();
extern void wrd_error();
extern void wrd_version();
extern wordlist *wrd_saves;

/* xinterface.c */

extern void xi_arc();
extern bool xi_init();
extern bool xi_dump();
extern void xi_clearscreen();
extern void xi_close();
extern void xi_drawline();
extern void xi_resetcolor();
extern void xi_setcolor();
extern void xi_setlinestyle();
extern void xi_text();
extern void xi_update();
extern void xi_zoomdata();
extern struct screen *screens;
extern void com_clearplot();

/* newcoms.c */
extern void com_reshape();

/* dimens.c */
extern void dimstring();
extern int atodims();
extern void indexstring();
extern int incindex( );

#endif /* FTEext_h */
