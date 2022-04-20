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
    fprintf(stderr, "OSDI(unkown) %s", handle->name);
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
