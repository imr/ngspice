/* Enhancement-200: Touchstone (.sNp) -> Verilog-A converter + `pre_snp` command. */
#ifndef SNP2VA_H
#define SNP2VA_H
#include "ngspice/wordlist.h"
/* Convert a Touchstone file to a Verilog-A n-port model. Returns 0 on success;
 * msg gets a one-line status/error. (No ngspice deps in the converter core.) */
int snp2va_convert(const char *snpfile, const char *vafile, const char *module,
                   char *msg, int msglen);
/* Enhancement-242: same parse+vector-fit, emitted as a native `.nport` fit file
 * (for the built-in n-port device) instead of Verilog-A. */
int snp2nport_convert(const char *snpfile, const char *nportfile,
                      char *msg, int msglen);
void com_pre_snp(wordlist *wl);
#endif
