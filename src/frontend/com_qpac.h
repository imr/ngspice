#ifndef ngspice_COM_QPAC_H
#define ngspice_COM_QPAC_H

/* Enhancement-137: two-tone small-signal QPAC (quasi-periodic AC). */
void com_qpac(wordlist *wl);

/* Enhancement-142: shared sweep helpers (dec/oct/lin classification, point count, and
 * the ngspice-plot emitter) used by the qpac/qpnoise/qpxf frequency sweeps. */
int  qp_steptype(const char *w);
int  qp_sweep_maxpts(int stepType, int np, double fstart, double fstop);
void qp_emit_plot(const char *plotname, const char *title, double *freqs, int npts,
                  char **vnames, int nvec, double *data);

#endif
