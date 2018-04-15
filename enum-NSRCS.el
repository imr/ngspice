;;; (load (buffer-file-name))

(require 'cl-lib)

(defun enum-NSRCS ()
  (goto-char (point-min))
  (while (re-search-forward (rx (*? (any "\n" space))
                                (group "/* finally, the number of noise sources */")
                                (*? (any "\n" space))
                                (group (*? alnum) "NSRCS")
                                (*? (any "\n" space))
                                "};"
                                (? (*? (any "\n" space))
                                   "/* the number of "
                                   (+? print)
                                   " noise sources" (*? space) "*/"
                                   (*? space))
                                )
                            nil t)
    (let ((p0 (copy-marker (match-beginning 0)))
          (p1 (copy-marker (match-end 0))))
      (replace-match "\n \\1\n \\2\n};")
      (indent-region p0 p1))))

(cl-loop with files = (process-lines "git" "grep"
                                     "-l" "-e" "NSRCS")
         for file in files
         do
         (progn
           (with-temp-file file
             (insert-file-contents file)
             (c-mode)
             (setq c-file-style  "BSD")
             (setq c-basic-offset 4)
             (enum-NSRCS))))

