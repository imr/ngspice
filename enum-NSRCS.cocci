// (compile "git grep -l -e 'NSRCS' -- '*.h' | xargs -n1 spatch --sp-file enum-NSRCS.cocci --in-place && emacs -q --no-site-file --batch --load enum-NSRCS.el && git commit -am 'execute enum-NSRCS.cocci'")
// (compile "spatch --sp-file enum-NSRCS.cocci src/spicelib/devices/jfet/jfetdefs.h")


@nsrcs@
identifier nsrcs =~ "NSRCS$";
@@
  enum {
    ...,
-   nsrcs,
+   /* finally, the number of noise sources */
+   nsrcs
  }
