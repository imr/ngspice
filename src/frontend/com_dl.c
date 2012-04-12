#include "ngspice/ngspice.h" /* for wl */
#include "ngspice/ftedefs.h"
#include "ngspice/devdefs.h" /* solve deps in dev.h*/
#include <../spicelib/devices/dev.h> /*for load library commands*/
#include "com_dl.h"

#if ADMS >= 3 || 1
void com_admsmodel(wordlist *wl){
    if(!wl || !wl->wl_next) {
        fprintf(cp_err,"Error: admsmodel, usage ...\n");
        return;
    }
    if(load_vadev_(wl->wl_word, wl->wl_next->wl_word))
        fprintf(cp_err,"Error: ADMS Library %s couldn't be loaded!\n",wl->wl_word);
    return;
}
#endif
#ifdef XSPICE
void com_codemodel(wordlist *wl){
  wordlist *ww;
  for(ww = wl;ww;ww = ww->wl_next)
    if(load_opus(wl->wl_word))
      fprintf(cp_err,"Error: Library %s couldn't be loaded!\n",ww->wl_word);
  return;
}
#endif
#ifdef DEVLIB
void com_use(wordlist *wl){
    wordlist *ww;  
  for(ww = wl;ww;ww = ww->wl_next)
    if(load_dev(wl->wl_word))
      fprintf(cp_err,"Error: Library %s couldn't be loaded!\n",ww->wl_word);
  return;
}
#endif

