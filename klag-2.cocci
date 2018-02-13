// DEVmodDelete()
//   drop the module search loop
//   drop the instance list loop
//   don't FREE the module struct
//
// (compile "git --no-pager grep -l -e 'Delete' -- '*.[ch]' ':(exclude)**/mifmdelete.c' ':(exclude)**/dlmain.c' | xargs -P8 -n1 spatch --sp-file klag-2.cocci --in-place && git commit -am 'DEVmodDelete(), change API, #1/2, coccinelle semantic patch'")
//
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/ndev/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/cap/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/bjt/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/asrc/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/bsim1/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/xspice/mif/*.[ch]")
// (compile "spatch --sp-file klag-2.cocci src/spicelib/devices/ndev/ndevmdel.c")

// in three steps, ...
@unify_DEVmodDelete@
identifier fn =~ "mDelete$";
typedef GENmodel, IFuid, GENinstance;
identifier XmodList, Xmodname, XkillModel;
type Tm, Ti;
identifier Xmodel, Xmodfast, Xhere, Xprev, Xoldmod;
identifier XXXnextModel, XXXmodName;
identifier Xlabel;
symbol gen_model;
symbol OK, E_NOMOD;
@@
  int fn (GENmodel **XmodList, IFuid Xmodname, GENmodel *XkillModel)
  {
(
-     Tm **Xmodel = (Tm **) XmodList;
-     Tm *Xmodfast = (Tm *) XkillModel;
|
-     Tm *Xmodfast = (Tm *) XkillModel;
-     Tm **Xmodel = (Tm **) XmodList;
)
(
-     Ti *Xhere;
-     Ti *Xprev = NULL;
|
)
-     Tm **Xoldmod;
-
-     Xoldmod = Xmodel;
-     for (; *Xmodel; Xmodel = &((*Xmodel)->XXXnextModel)) {
-          if ((*Xmodel)->XXXmodName == Xmodname || (Xmodfast && *Xmodel == Xmodfast)) {
-              goto Xlabel;
-          }
-          Xoldmod = Xmodel;
-     }
-
-     return(E_NOMOD);
+     NG_IGNORE(gen_model);
      ...
}


@unify_DEVmodDelete_1 extends unify_DEVmodDelete@
identifier XXXnextInstance =~ "nextInstance$";
identifier XXXdelete =~ "delete$";
identifier XXXinstances;
@@
  int fn (GENmodel **XmodList, IFuid Xmodname, GENmodel *XkillModel)
  {
      NG_IGNORE(gen_model);
-Xlabel:
(
-     if ((*Xmodel)->XXXinstances) {
-         return(E_NOTEMPTY);
-     }
|
)
-     *Xoldmod = (*Xmodel)->XXXnextModel;
(
-     for (Xhere = ...; Xhere; Xhere = Xhere->XXXnextInstance) {
-         if (Xprev) {
?-            XXXdelete((GENinstance *) Xprev);
-             FREE(Xprev);
-         }
-         Xprev = Xhere;
-     }
-     if (Xprev) {
?-        XXXdelete((GENinstance *) Xprev);
-         FREE(Xprev);
-     }
|
)
-     FREE(*Xmodel);
-     return(OK);
  }


@unify_DEVmodDelete_2 extends unify_DEVmodDelete@
@@
int fn (
-  GENmodel **XmodList, IFuid Xmodname, GENmodel *XkillModel
+  GENmodel *gen_model
  )
{
     NG_IGNORE(gen_model);
+    return OK;
}


// deal with those which are alreay empty
@unify_DEVdelete_empty@
identifier fn =~ "mDelete$";
identifier XmodList, Xmodname, XkillModel;
@@
  int fn (
-  GENmodel **XmodList, IFuid Xmodname, GENmodel *XkillModel
+  GENmodel *gen_model
  )
  {
-     <+... NG_IGNORE(...); ...+>
-     return(OK);
+     NG_IGNORE(gen_model);
+     return OK;
  }


// deal with prototypes
@unify_DEVmodDelete_h@
identifier fn =~ "mDelete$";
@@
  extern int fn (
-   GENmodel **, IFuid,
    GENmodel *
  );


