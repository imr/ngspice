;;; (compile "emacs -q --no-site-lisp --script tmp-empty-fun.el src/spicelib/devices/mos1/mos1dest.c")

(defun drop(file base)
    (with-temp-file file
      (insert-file-contents-literally file)
      (while (re-search-forward (concat (rx bol (* any) bow)
                                        (regexp-quote base)
                                        (rx eow (* any) "\n"))
                                nil t)
        (replace-match ""))))

(defun moan(file)
  (unless (string-match "ChangeLog" file)
    (message "remnant found in %s" file)))


(let* ((file (car argv))
       (dir (file-name-directory file))
       (base (file-name-nondirectory file)))
  (message "processing %s" file)
  (mapcar (lambda (process-file)
            (drop process-file base))
          (ignore-errors
            (process-lines "git" "grep" "-l" base
                           "--" dir "*.vcxproj")))
  (mapcar #'moan
          (ignore-errors
            (process-lines "git" "grep" "-l" (file-name-base file)))))
