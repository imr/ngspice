#include "ngspice/devdefs.h"
#include "osdidefs.h"

void osdi_log(void *handle_, char *msg, uint32_t lvl) {
  OsdiNgspiceHandle *handle = handle_;
  FILE *dst = stdout;
  switch (lvl & LOG_LVL_MASK) {
  case LOG_LVL_DEBUG:
    printf("OSDI(debug) %s: ", handle->name);
    break;
  case LOG_LVL_DISPLAY:
    printf("OSDI %s: ", handle->name);
    break;
  case LOG_LVL_INFO:
    printf("OSDI(info) %s: ", handle->name);
    break;
  case LOG_LVL_WARN:
    fprintf(stderr, "OSDI(warn) %s: ", handle->name);
    dst = stderr;
    break;
  case LOG_LVL_ERR:
    fprintf(stderr, "OSDI(err) %s: ", handle->name);
    dst = stderr;
    break;
  case LOG_LVL_FATAL:
    fprintf(stderr, "OSDI(fatal) %s: ", handle->name);
    dst = stderr;
    break;
  default:
    fprintf(stderr, "OSDI(unknown) %s", handle->name);
    break;
  }

  if (lvl & LOG_FMT_ERR) {
    fprintf(dst, "failed to format\"%s\"\n", msg);
  } else {
    fprintf(dst, "%s", msg);
  }
}

double osdi_pnjlim(bool init, bool *check, double vnew, double vold, double vt,
                   double vcrit) {
  if (init) {
    *check = true;
    return vcrit;
  }
  int icheck = 0;
  double res = DEVpnjlim(vnew, vold, vt, vcrit, &icheck);
  *check = icheck != 0;
  return res;
}

double osdi_limvds(bool init, bool *check, double vnew, double vold) {
  if (init) {
    *check = true;
    return 0.1;
  }
  double res = DEVlimvds(vnew, vold);
  if (res != vnew) {
    *check = true;
  }
  return res;
}

double osdi_fetlim(bool init, bool *check, double vnew, double vold,
                   double vto) {
  if (init) {
    *check = true;
    return vto + 0.1;
  }
  double res = DEVfetlim(vnew, vold, vto);
  if (res != vnew) {
    *check = true;
  }
  return res;
}

double osdi_limitlog(bool init, bool *check, double vnew, double vold,
                     double LIM_TOL) {
  if (init) {
    *check = true;
    return 0.0;
  }
  int icheck = 0;
  double res = DEVlimitlog(vnew, vold, LIM_TOL, &icheck);
  *check = icheck != 0;
  return res;
}

/* OSDI v0.5 (2C) — absdelay exact-transport reader.  Bound to
 * SimInfo.delay_read; the model calls it mid-eval to obtain
 * x(abstime - td) by linear interpolation over the per-instance ring
 * (accepted (time,value) samples, oldest..newest).  SimInfo.delay_state
 * points at the instance's OsdiExtraInstData so the ring is reachable
 * without a handle (thread-safe: SimInfo is per-eval).  max_td < 0 means
 * "unbounded"; otherwise td is clamped to max_td (matching absdelay's
 * max_delay clamp).  For a target time newer than the last accepted
 * sample (td smaller than the current local step) we interpolate toward
 * the current eval's input (delay_input_arr[site], just written by the
 * model this eval), so small delays are exact rather than held. */
double osdi_delay_read(const OsdiSimInfo *info, uint32_t site, double td,
                       double max_td) {
  OsdiExtraInstData *extra = (OsdiExtraInstData *)info->delay_state;
  if (!extra || !extra->delay_t || !extra->delay_v || !extra->delay_count)
    return 0.0;
  if (td < 0.0)
    td = 0.0;
  if (max_td >= 0.0 && td > max_td)
    td = max_td;

  /* OSDI v0.5 (2C AC delay) — stash the (clamped) td so OSDIacLoad can read
   * td(OP) for this site's e^{-jw*td} delay jacobian. */
  if (extra->delay_td_arr)
    extra->delay_td_arr[site] = td;

  uint32_t n = extra->delay_count[site];
  double *t = extra->delay_t[site];
  double *v = extra->delay_v[site];
  double t_target = info->abstime - td;

  if (n == 0)
    return extra->delay_input_arr ? extra->delay_input_arr[site] : 0.0;

  /* Older than the oldest accepted sample: hold the oldest value. */
  if (t_target <= t[0])
    return v[0];

  /* Newer than the last accepted sample: interpolate toward the current
   * eval's input at abstime. */
  if (t_target >= t[n - 1]) {
    double t_now = info->abstime;
    double v_now = extra->delay_input_arr ? extra->delay_input_arr[site]
                                          : v[n - 1];
    double span = t_now - t[n - 1];
    if (span <= 0.0)
      return v[n - 1];
    double frac = (t_target - t[n - 1]) / span;
    if (frac > 1.0)
      frac = 1.0;
    return v[n - 1] + frac * (v_now - v[n - 1]);
  }

  /* Find the bracket [t[i], t[i+1]] with t[i] <= t_target < t[i+1],
   * scanning back from the newest (delays are usually a few steps). */
  uint32_t i = n - 1;
  while (i > 0 && t[i] > t_target)
    i--;
  double span = t[i + 1] - t[i];
  if (span <= 0.0)
    return v[i];
  double frac = (t_target - t[i]) / span;
  return v[i] + frac * (v[i + 1] - v[i]);
}
