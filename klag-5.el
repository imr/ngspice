;;; (load (buffer-file-name))

(require 'cl-lib)

(defun mx-klag-5 ()
  (goto-char (point-min))
  (while (re-search-forward (rx bol (*? space)
                                "/* emacs-place-here */"
                                ;; inclusive comments
                                (* (* (or space
                                       (: "/*" (*? (or (not (any "/*"))
                                                       (: "*" (not (any "/")))
                                                       (: "/" (not (any "/*")))))
                                          "*/")))
                                   "\n"))
                            nil t)
    (let ((p (match-beginning 0)))
      (replace-match "\n")
      (save-excursion
        (save-match-data
          (when (re-search-forward (rx bol (*? space)
                                       "/* emacs-from-here */"
                                       (group (+? anything))
                                       "/* emacs-upto-here */"
                                       (*? space) "\n")
                                   nil t)
            (let ((s (match-string 1)))
              (replace-match "")
              (goto-char p)
              (insert s))))))))


(defun mx-klag-5+ ()
  "mop up some whitespace from cocci"
  (goto-char (point-min))
  (while (re-search-forward (rx bol (* (*? space) "\n")
                                (group
                                 (*? space)
                                 (or "struct GENinstance gen"
                                     "struct GENmodel gen")))
                            nil t)
    (replace-match "\n\\1")))


(loop
   with files = (process-lines "git" "grep" "-l"
                               "-e" "emacs-place-here"
                               "--" "*.[ch]")
   for file in files
   do
     (progn
       (with-temp-file file
         (insert-file-contents file)
         (c-mode)
         (setq c-file-style  "BSD")
         (setq c-basic-offset 4)
         (mx-klag-5)
         (mx-klag-5+))))
