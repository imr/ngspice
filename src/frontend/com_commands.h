#ifndef ngspice_COM_COMMANDS_H
#define ngspice_COM_COMMANDS_H

void com_showmod(wordlist *wl);
void com_show(wordlist *wl);
void com_alter(wordlist *wl);
void com_altermod(wordlist *wl);
void com_alterparam(wordlist *wl);
void com_optimize(wordlist *wl);   /* Enhancement-130 */
void com_rfstab(wordlist *wl);     /* Enhancement-253 */
void com_loadpull(wordlist *wl);   /* Enhancement-234 */
void com_qpss(wordlist *wl);       /* Enhancement-133 */
void com_qpac(wordlist *wl);       /* Enhancement-137 */
void com_qpxf(wordlist *wl);       /* Enhancement-141 */
void com_qpnoise(wordlist *wl);    /* Enhancement-138 */
void com_hbosc(wordlist *wl);      /* Enhancement-140 */
void com_phasenoise(wordlist *wl); /* Enhancement-140 */
void com_hb(wordlist *wl);         /* Enhancement-134 */
void com_savestate(wordlist *wl);  /* Enhancement-131 */
void com_loadstate(wordlist *wl);  /* Enhancement-131 */
void com_meas(wordlist *wl);
void com_sysinfo(wordlist *wl);
void com_check_ifparm(wordlist *wl);

#endif
