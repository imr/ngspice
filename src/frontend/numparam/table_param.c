/*
 * HSPICE table_param() — see table_param.h for syntax.
 *
 * Implementation strategy:
 *
 *   1. Parse each .table file once on first reference.  Stored as a
 *      flat array of rows (key + value doubles per row).  The header
 *      column count tells us how many doubles per row.  We don't
 *      pre-sort or index rows — typical PDK tables are <1000 rows so
 *      a linear filter is fine, AND the keys-then-interpolation
 *      structure makes row order matter (rows are pre-sorted by the
 *      foundry tool with the real-key column ascending, so we can
 *      find bracketing rows by simple scan).
 *
 *   2. Lookup: filter rows where ALL integer keys exactly match (the
 *      n_int leftmost columns).  Among matching rows, linear-interpolate
 *      over the real-key columns (n_real columns immediately after the
 *      integer keys) using the requested point.
 *
 *   3. Cache: NGHASHPTR from resolved-filename -> Table*.  Cache lives
 *      for process lifetime.  Foundry PDKs typically reference ~20
 *      distinct .table files, each ~hundreds of KB; total memory cost
 *      a few MB tops.
 *
 * Interpolation: for a single real key, this is straight linear
 * interpolation between the two bracketing rows.  For multiple real
 * keys, we use the obvious tensor-product extension — find the
 * bracketing range in EACH real-key dimension, then iterate over the
 * 2^n_real corners and weight them by the multilinear factors.  Out-
 * of-range real values clamp to the table's edge value (no
 * extrapolation, matches HSPICE's default).
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/hash.h"
#include "ngspice/fteinp.h"
#include "ngspice/stringutil.h"
#include "../inpcom.h"
#include "table_param.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

typedef struct Table {
    int n_cols;           /* total columns including keys */
    int n_rows;
    double *data;         /* row-major: data[row * n_cols + col] */
} Table;

static NGHASHPTR g_table_cache = NULL;

/* ---------------------------------------------------------------- */
/* File parsing.                                                    */
/* ---------------------------------------------------------------- */

/* Count whitespace-separated tokens on a line.  Stops at end of
 * string or '\n'.  Used once to size the column count from the
 * header. */
static int count_tokens(const char *s) {
    int n = 0;
    while (*s) {
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s || *s == '\n') break;
        n++;
        while (*s && !isspace((unsigned char)*s)) s++;
    }
    return n;
}

/* Parse one whitespace-separated double from `*p`, advance `*p` past
 * it.  Returns true on success, false on EOL/EOF or parse error. */
static bool parse_double_token(const char **p, double *out) {
    const char *s = *p;
    while (*s && isspace((unsigned char)*s) && *s != '\n') s++;
    if (!*s || *s == '\n' || *s == '\0') return false;
    char *end;
    double v = strtod(s, &end);
    if (end == s) return false;
    *out = v;
    *p = end;
    return true;
}

/* Free a parsed table.  Callback signature matches nghash_free's
 * del_data slot. */
static void free_table(void *p) {
    Table *t = (Table *)p;
    if (!t) return;
    free(t->data);
    free(t);
}

/* Read the file at `path`, parse the header to learn column count,
 * then parse data rows.  Skip lines starting with '*' or '#' (the
 * latter only after the first one — the first '#' line IS the
 * header).  Returns the allocated Table on success, NULL on
 * failure (with error reported via fprintf). */
static Table *load_table_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr,
                "Error: table_param: cannot open table file '%s'\n",
                path);
        return NULL;
    }

    /* First non-blank line MUST be the header (starting with '#').
     * Count its tokens (excluding the leading '#') to learn n_cols. */
    char buf[8192];
    int n_cols = -1;
    while (fgets(buf, sizeof buf, fp)) {
        char *p = buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == '\n') continue;
        if (*p == '#') {
            p++;  /* skip the # */
            n_cols = count_tokens(p);
            break;
        }
        /* No header — treat first line as data, count its tokens.
         * Then we'll need to re-read it.  Simpler: insist on a header. */
        fprintf(stderr,
                "Error: table_param: file '%s' missing '#'-header "
                "line\n",
                path);
        fclose(fp);
        return NULL;
    }
    if (n_cols <= 0) {
        fprintf(stderr,
                "Error: table_param: file '%s' header has 0 columns\n",
                path);
        fclose(fp);
        return NULL;
    }

    /* Read all subsequent data rows.  Grow buffer as needed. */
    int cap = 256;
    int n_rows = 0;
    double *data = (double *)malloc(sizeof(double) * cap * n_cols);
    if (!data) {
        fprintf(stderr, "Error: table_param: out of memory\n");
        fclose(fp);
        return NULL;
    }

    while (fgets(buf, sizeof buf, fp)) {
        const char *p = buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == '\n' || *p == '*' || *p == '#') continue;

        if (n_rows >= cap) {
            cap *= 2;
            double *nd = (double *)realloc(
                    data, sizeof(double) * cap * n_cols);
            if (!nd) {
                free(data);
                fprintf(stderr, "Error: table_param: out of memory\n");
                fclose(fp);
                return NULL;
            }
            data = nd;
        }

        double *row = &data[n_rows * n_cols];
        int c;
        for (c = 0; c < n_cols; c++) {
            if (!parse_double_token(&p, &row[c])) {
                fprintf(stderr,
                        "Error: table_param: file '%s' row %d "
                        "has fewer than %d columns\n",
                        path, n_rows + 1, n_cols);
                free(data);
                fclose(fp);
                return NULL;
            }
        }
        n_rows++;
    }

    fclose(fp);

    Table *t = (Table *)malloc(sizeof(Table));
    if (!t) {
        free(data);
        fprintf(stderr, "Error: table_param: out of memory\n");
        return NULL;
    }
    t->n_cols = n_cols;
    t->n_rows = n_rows;
    t->data = data;
    return t;
}

/* ---------------------------------------------------------------- */
/* Cache.                                                           */
/* ---------------------------------------------------------------- */

/* Resolve a relative path via ngspice's sourcepath search.  Returns
 * a malloc'd absolute path the caller must free, or a copy of the
 * input if no search is needed.
 *
 * Foundry PDKs (Samsung 14LPU) reference tables by paths relative to
 * the .lib file that uses them — `./RF_COMPONENTS/foo.table` is
 * relative to the directory of fets_rf.lib (or wherever the call
 * originated), NOT to the user's cwd or ngspice's sourcepath.
 * ngspice doesn't track per-card include-directory currently, so:
 *   1. Try the path as-is (works for absolute paths or if user cd'd
 *      into the PDK directory).
 *   2. Walk ngspice's `sourcepath` list (inp_pathresolve handles this)
 *      — works if the user added the PDK's NGSPICE dir to sourcepath.
 *   3. Try the directory of the most recently included file
 *      (`inputdir` global).
 * If none of those work, the caller will get a clear file-not-found
 * error and can add the PDK directory to `set sourcepath`. */
extern char *inputdir;  /* set by inp_spsource during deck loading */

/* Try to find `filename` relative to `dir`.  Returns malloc'd
 * absolute path on success, NULL on failure.  `dir` is the
 * directory containing the .lib file that emitted the call. */
static char *try_in_dir(const char *filename, const char *dir) {
    if (!dir || !dir[0])
        return NULL;
    size_t need = strlen(dir) + 1 + strlen(filename) + 1;
    char *combined = (char *)malloc(need);
    if (!combined) return NULL;
    snprintf(combined, need, "%s/%s", dir, filename);
    char *resolved = inp_pathresolve(combined);
    free(combined);
    return resolved;
}

/* Strip the trailing filename component from `path` so the result
 * is the directory.  Caller frees. */
static char *dirname_of(const char *path) {
    if (!path) return NULL;
    const char *slash = strrchr(path, '/');
    if (!slash) return copy(".");  /* no separator → cwd */
    size_t n = (size_t)(slash - path);
    char *r = (char *)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, path, n);
    r[n] = '\0';
    return r;
}

/* Resolve a relative .table path.  Search order:
 *   1. Absolute path: use as-is.
 *   2. As-given (cwd-relative or sourcepath-listed).
 *   3. Relative to `dir_hint` (caller passes the dirname of the
 *      .lib file containing the table_param call — matches HSPICE).
 *   4. Relative to ngspice's `inputdir` (the netlist file's dir).
 *   5. Bare filename — fopen will fail with a clear error. */
static char *resolve_table_path(const char *filename,
                                const char *dir_hint) {
    /* 1. Absolute. */
    if (filename[0] == '/')
        return copy(filename);

    /* 2. ngspice sourcepath / cwd. */
    char *resolved = inp_pathresolve(filename);
    if (resolved)
        return resolved;

    /* 3. Relative to the file the call came from (the typical case
     * for foundry PDKs: `./RF_COMPONENTS/...` inside fets_rf.lib
     * is relative to fets_rf.lib's directory). */
    resolved = try_in_dir(filename, dir_hint);
    if (resolved)
        return resolved;

    /* 4. Relative to inputdir (set by inp_spsource for the netlist
     * itself). */
    resolved = try_in_dir(filename, inputdir);
    if (resolved)
        return resolved;

    /* 5. Last resort. */
    return copy(filename);
}

static Table *get_or_load_table(const char *filename,
                                const char *dir_hint) {
    if (!g_table_cache)
        g_table_cache = nghash_init(8);

    Table *t = (Table *)nghash_find(g_table_cache, (void *)filename);
    if (t)
        return t;

    char *path = resolve_table_path(filename, dir_hint);
    t = load_table_file(path);
    tfree(path);

    if (!t)
        return NULL;

    /* Key the cache by an owned copy of the original filename string
     * (the caller's pointer may not persist). */
    char *key = copy(filename);
    nghash_insert(g_table_cache, key, t);
    return t;
}

void table_param_clear_cache(void) {
    if (!g_table_cache) return;
    nghash_free(g_table_cache, free_table, free);
    g_table_cache = NULL;
}

/* ---------------------------------------------------------------- */
/* Lookup.                                                          */
/* ---------------------------------------------------------------- */

/* Returns 1 if integer keys at this row match query (within
 * floating-point tolerance), 0 otherwise. */
static int row_int_keys_match(const Table *t, int row,
                              const double *int_vals, int n_int) {
    const double *r = &t->data[row * t->n_cols];
    for (int i = 0; i < n_int; i++) {
        if (fabs(r[i] - int_vals[i]) > 0.5)
            return 0;
    }
    return 1;
}

/* For the subset of rows where integer keys match, linearly
 * interpolate the output_col value over the n_real real keys.
 *
 * Implementation: this is the tensor-product multilinear
 * interpolation.  For each real-key dimension d in 0..n_real-1,
 * find the two distinct values v_lo, v_hi present in the matching
 * rows that bracket real_vals[d].  Compute a weight w_d in [0,1]
 * such that real_vals[d] = (1-w_d)*v_lo + w_d*v_hi.  Then the
 * interpolated output is the weighted sum over the 2^n_real corners
 * of the bracketing hyper-rectangle.
 *
 * Out-of-range real values clamp to the nearest available endpoint
 * (no extrapolation).
 */
static int interpolate_output(const Table *t,
                              const double *int_vals, int n_int,
                              const double *real_vals, int n_real,
                              int output_col,
                              double *out)
{
    /* output_col is 1-based, indexed into value columns (after the
     * keys).  Convert to absolute column index. */
    int data_col = n_int + n_real + (output_col - 1);
    if (data_col < n_int + n_real || data_col >= t->n_cols) {
        fprintf(stderr,
                "Error: table_param: output column %d out of range "
                "(table has %d value columns)\n",
                output_col, t->n_cols - n_int - n_real);
        return 1;
    }

    /* For each real-key dimension, collect the distinct values
     * present in matching rows, sorted. */
    int found_any = 0;
    double v_lo[16], v_hi[16];   /* up to 16 real keys; foundry uses ≤3 */
    double w[16];
    if (n_real > 16) {
        fprintf(stderr,
                "Error: table_param: too many real keys (%d, max 16)\n",
                n_real);
        return 1;
    }

    for (int d = 0; d < n_real; d++) {
        double target = real_vals[d];
        /* Find bracketing values: max v ≤ target and min v ≥ target,
         * among rows whose integer keys match AND whose previous real
         * keys match the previously-bracketed indices.  For simplicity
         * we do a per-dimension scan without the inter-dimensional
         * constraint — the foundry tables are products of grids in
         * each key dim, so this gives the correct result. */
        double best_lo = -INFINITY, best_hi = INFINITY;
        int have_lo = 0, have_hi = 0;
        for (int r = 0; r < t->n_rows; r++) {
            if (!row_int_keys_match(t, r, int_vals, n_int))
                continue;
            double v = t->data[r * t->n_cols + n_int + d];
            if (v <= target && v > best_lo) {
                best_lo = v;
                have_lo = 1;
            }
            if (v >= target && v < best_hi) {
                best_hi = v;
                have_hi = 1;
            }
            found_any = 1;
        }
        if (!have_lo && !have_hi) {
            fprintf(stderr,
                    "Error: table_param: no matching rows for "
                    "integer keys\n");
            return 1;
        }
        /* Clamp at edges */
        if (!have_lo) best_lo = best_hi;
        if (!have_hi) best_hi = best_lo;
        v_lo[d] = best_lo;
        v_hi[d] = best_hi;
        if (best_hi == best_lo)
            w[d] = 0.0;
        else
            w[d] = (target - best_lo) / (best_hi - best_lo);
        if (w[d] < 0.0) w[d] = 0.0;
        if (w[d] > 1.0) w[d] = 1.0;
    }

    if (!found_any) {
        fprintf(stderr,
                "Error: table_param: no matching rows found\n");
        return 1;
    }

    /* Iterate over the 2^n_real corners.  For each corner, find the
     * row whose real-key columns match v_lo[d] or v_hi[d] per the
     * corner's bits, and accumulate weighted output. */
    double sum = 0.0;
    double corner_weight_sum = 0.0;
    int n_corners = 1 << n_real;
    for (int c = 0; c < n_corners; c++) {
        /* Build the real-key target for this corner. */
        double rk[16];
        double weight = 1.0;
        for (int d = 0; d < n_real; d++) {
            if (c & (1 << d)) {
                rk[d] = v_hi[d];
                weight *= w[d];
            } else {
                rk[d] = v_lo[d];
                weight *= (1.0 - w[d]);
            }
        }
        if (weight == 0.0)
            continue;

        /* Find the matching row. */
        int found = 0;
        for (int r = 0; r < t->n_rows; r++) {
            if (!row_int_keys_match(t, r, int_vals, n_int))
                continue;
            int rk_match = 1;
            for (int d = 0; d < n_real; d++) {
                double v = t->data[r * t->n_cols + n_int + d];
                if (fabs(v - rk[d]) > 1e-12 * (fabs(rk[d]) + 1e-30)) {
                    rk_match = 0;
                    break;
                }
            }
            if (!rk_match) continue;
            sum += weight * t->data[r * t->n_cols + data_col];
            corner_weight_sum += weight;
            found = 1;
            break;
        }
        if (!found) {
            /* Sparse table: a bracketing corner row doesn't exist.
             * The output will still be a useful approximation as
             * long as at least one corner was found. */
        }
    }

    if (corner_weight_sum == 0.0) {
        fprintf(stderr,
                "Error: table_param: interpolation failed "
                "(no bracketing rows found)\n");
        return 1;
    }

    /* Normalize in case some corners were missing — preserves the
     * weighted-average interpretation. */
    *out = sum / corner_weight_sum;
    return 0;
}

int table_param_lookup(const char *filename,
                       const char *dir_hint,
                       const double *int_vals, int n_int,
                       const double *real_vals, int n_real,
                       int output_col,
                       double *out)
{
    Table *t = get_or_load_table(filename, dir_hint);
    if (!t)
        return 1;

    if (t->n_cols < n_int + n_real + 1) {
        fprintf(stderr,
                "Error: table_param: file '%s' has %d columns but "
                "call uses %d keys + at least 1 value column\n",
                filename, t->n_cols, n_int + n_real);
        return 1;
    }

    return interpolate_output(t, int_vals, n_int,
                              real_vals, n_real,
                              output_col, out);
}
