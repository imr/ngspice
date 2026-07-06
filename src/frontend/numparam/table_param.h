/*
 * HSPICE table_param() implementation.
 *
 * Foundry PDKs (Samsung 14LPU, TSMC, GF) use HSPICE's table_param()
 * extensively for table-file-based parameter lookup — self-heating
 * thermal resistance, RF parasitics, etc.  Samsung's TT corner alone
 * references it ~1300 times.
 *
 * Syntax:
 *     table_param(file, N_int_keys, int_val1, ..., int_valN,
 *                       N_real_keys, real_val1, ..., real_valM,
 *                       output_col)
 *
 *   file        — path to .table file (relative paths resolve via
 *                 ngspice sourcepath)
 *   N_int_keys  — number of integer-typed search keys (exact match)
 *   int_valX    — integer key values to match
 *   N_real_keys — number of real-typed search keys (linear-interpolated)
 *   real_valX   — real key values to interpolate between
 *   output_col  — 1-based index into the *value* columns of the table
 *                 (i.e., columns after the keys)
 *
 * Table file format:
 *     #key1 key2 ... keyN val1 val2 ... valM    (header comment)
 *     k1 k2 ... kN v1 v2 ... vM                  (data rows)
 *
 * The header line names the columns; the first N columns are keys
 * (matching the call's int + real keys in declared order); the
 * remaining M columns are output values selected by output_col.
 *
 * Files are loaded on first reference and cached for the lifetime
 * of the process (foundry PDKs reference the same table file
 * hundreds of times).
 */

#ifndef TABLE_PARAM_H
#define TABLE_PARAM_H

#include <stdbool.h>

/* Evaluate a table_param() call.
 *
 *   filename       — table file path (caller's string, not retained)
 *   int_vals       — array of integer key values; length n_int
 *   n_int          — number of integer keys
 *   real_vals      — array of real key values; length n_real
 *   n_real         — number of real keys
 *   output_col     — 1-based index into value columns
 *   out            — written with the looked-up (interpolated) value
 *                    on success; unchanged on failure
 *
 * Returns 0 on success, non-zero on error (file not found, key out
 * of range, output column out of range, etc.).  Errors are reported
 * via fprintf(stderr, ...).
 */
/* `dir_hint` is the directory of the file that originated this
 * call (typically the .lib file containing the table_param()
 * invocation).  Relative `filename` paths resolve against this dir
 * first, matching HSPICE's behavior.  May be NULL — falls back to
 * cwd / sourcepath / ngspice's `inputdir`. */
int table_param_lookup(const char *filename,
                       const char *dir_hint,
                       const double *int_vals, int n_int,
                       const double *real_vals, int n_real,
                       int output_col,
                       double *out);

/* Free all cached tables.  Called at simulator shutdown. */
void table_param_clear_cache(void);

#endif /* TABLE_PARAM_H */
