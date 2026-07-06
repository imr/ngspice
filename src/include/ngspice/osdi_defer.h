/*
 * OSDI deferred-evaluation side table.
 *
 * HSPICE-style PDK model cards (Samsung 14LPU et al.) embed expressions
 * like
 *     cgbn={((l<=1e-07)*(1e-012)+(l>1e-07)*(...))}
 * directly on the right-hand side of `.model` parameters.  The expression
 * references `l`/`w`/`nf` etc. — per-instance geometry that is not defined
 * at .model parse time.  HSPICE defers evaluation until each instance is
 * bound and re-evaluates with the instance's geometry in scope.
 *
 * ngspice's numparam evaluates `{...}` expressions eagerly when the
 * `.model` line is processed, which fails ("Undefined parameter [l]")
 * before any instance exists.  This side table fixes that path for OSDI
 * models only:
 *
 *   - At numparam time, when we see a `{...}` on a `.model` line that is
 *     headed for an OSDI device type AND that contains per-instance
 *     symbols, we save the expression text here (keyed by model + param
 *     name) and replace the brace block with the literal `0` so numparam
 *     can finish the line successfully.  The model gets a sane (if
 *     placeholder) default for the parameter.
 *
 *   - At instance creation, osdiload.c walks the saved entries for this
 *     instance's model, evaluates each expression with the instance's
 *     `l`/`w`/`nf`/`m`/`xnf` pushed into a transient numparam scope, and
 *     calls OSDIparam to install the result as a per-instance override of
 *     the model default.
 *
 * Built-in MOSFETs (BSIM3/4, HiSIM, etc.) are unaffected — their model
 * cards historically use pure-numeric values (with binning), so they
 * never hit the deferred path.
 */

#ifndef NGSPICE_OSDI_DEFER_H
#define NGSPICE_OSDI_DEFER_H

#include "ngspice/ngspice.h"

/* Per-(model, param) deferred-expression entry.  Stored in a singly-linked
 * list per model name; lookups happen at instance bind time.  Both
 * `model_name` and `param_name` are case-insensitively compared.
 *
 * `snap_*` capture the subckt-instance scope visible at register time.
 * HSPICE-style PDKs put `.model` cards INSIDE a `.subckt` body and let
 * the model's expressions reference subckt-scope `.params` (e.g.
 * Samsung 14LPU's `vsat1=...*(1+vsat_nfet/...)*velsat_mult` where
 * `vsat_nfet`/`velsat_mult`/`xl_nfet` are passed to the subckt via
 * its `params:` list).  By register time the subckt scope has been
 * resolved per-instance (subckt expansion produced model names like
 * `x2.xmn1:nfet.0`), so we capture the scalar values of every
 * non-instance-geom identifier referenced by the expression.  At
 * eval time these are pushed alongside l/w/nf/m so the expression
 * can resolve.  Without this, every subckt-scope reference fails
 * with `Undefined parameter` and the deferred parameter ends up 0. */
typedef struct osdi_defer_entry_s {
    char *model_name;          /* duplicated, freed at exit */
    char *param_name;          /* duplicated */
    char *expr_text;           /* duplicated; the raw { ... } body */
    int   n_snap;              /* count of snapshotted scope bindings */
    char **snap_names;         /* malloc'd array, each entry duplicated */
    double *snap_values;       /* malloc'd array, parallel to snap_names */
    struct osdi_defer_entry_s *next;
} OsdiDeferEntry;

/* Register a deferred expression.  Returns OK / non-zero on alloc fail.
 * Strings are duplicated internally; caller owns its inputs. */
int osdi_defer_register(const char *model_name,
                        const char *param_name,
                        const char *expr_text);

/* Iterate all entries for a given model name; calls `cb` once per entry
 * with the (param_name, expr_text, cb_arg).  Returns the number of entries
 * iterated. */
int osdi_defer_for_model(const char *model_name,
                         void (*cb)(const char *param_name,
                                    const char *expr_text,
                                    void *cb_arg),
                         void *cb_arg);

/* Returns true if any deferred entries exist for the given model name.
 * Avoids the iteration overhead at instance-bind time when there's
 * nothing to evaluate. */
bool osdi_defer_has(const char *model_name);

/* Free the entire side table.  Called at simulator shutdown. */
void osdi_defer_clear(void);

/* Record (lmin, lmax) for a bin-selected model name.  Called from
 * INPgetModBin once the model line's lmin/lmax tokens have been
 * parsed, so that OSDIsetup's pre-eval pass can pick a default
 * L within the bin's range (midpoint).  Without this, default
 * L=30nm makes Samsung-PDK expressions like
 * `vsat1='(l==14n)*X + (l==16n)*Y'` evaluate to 0 and BSIM-CMG
 * rejects "vsat1 = 0".  Matched against the runtime model name
 * (after subckt-path prefix is stripped). */
void osdi_defer_record_bin_range(const char *model_name,
                                 double lmin, double lmax);

/* Look up the recorded (lmin, lmax) for a model name.  Returns true
 * on hit (fills *lmin, *lmax), false otherwise.  Used by OSDIsetup
 * to set the pre-eval default L per-model. */
bool osdi_defer_get_bin_range(const char *model_name,
                              double *lmin, double *lmax);

/* Preprocess a raw `.model` line.  If the line targets an OSDI-level
 * device (level=72 / level=77) and contains `{...}` or `'...'`
 * expressions that reference per-instance symbols (l, w, nf, m, xnf),
 * extract each such expression into the side table keyed by
 * (model_name, param_name) and rewrite the expression in the line to
 * the literal `0` so the downstream numparam pass sees a clean
 * numeric value.  The model parameter then gets a placeholder 0; the
 * actual value is computed per-instance at bind time by walking the
 * side table.
 *
 * `*line_p` must point to a heap-allocated line buffer; this function
 * may free it and replace with a freshly-allocated buffer of the
 * rewritten line.  Returns 0 on success, non-zero on alloc error
 * (line is left unchanged).  No-op (returns 0, leaves line intact) for
 * non-`.model` lines or non-OSDI levels. */
int osdi_defer_preprocess_line(char **line_p);

/* Evaluate a deferred expression with per-instance geometry in scope.
 * Names `l`, `w`, `nf`, `m`, `xnf` are bound to the four numeric args
 * (xnf aliases nf — HSPICE convention).  Returns 0 / writes scalar
 * result to *out on success, non-zero on parse/eval error.  Thin
 * wrapper over nupa_eval_with_scope in the numparam engine. */
int osdi_defer_eval(const char *expr_text,
                    double l, double w, double nf, double m,
                    double *out);

#endif /* NGSPICE_OSDI_DEFER_H */
