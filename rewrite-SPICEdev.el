(defun mx-narrow-body (pos-0 pos-brace)
  (narrow-to-region pos-0
                    (save-excursion
                      (goto-char pos-brace)
                      (forward-sexp)
                      (point)))
  (goto-char (point-min)))

(defun fix(file)
  (with-temp-file file
    (insert-file-contents file)
    (c-mode)
    (setq c-file-style  "BSD")
    (setq c-basic-offset 4)
    (while (re-search-forward (rx bol
                                 (* space) "SPICEdev"
                                 (+ space) (+ alnum) (* space) "="
                                 (* space) (group "{"))
                             nil t)
      (let* ((a (match-beginning 0)))
        (set-mark
         (save-excursion
           (goto-char (match-beginning 1))
           (forward-sexp)
           (point)))
        (untabify a (mark))
        (save-excursion
          (while (re-search-forward
                  (rx (or (seq "//" (* any))
                          (seq "/*" (*? anything) "*/")))
                  (mark) t)
            (replace-match "")))
        (save-excursion
          (while (re-search-forward
                  (rx "}" (* space) "\n"
                      (* space) ",")
                  (mark) t)
            (replace-match "},")))
        (save-excursion
          (while (re-search-forward
                  (rx bol (+ (* space) "\n"))
                  (mark) t)
            (replace-match "")))
        (delete-trailing-whitespace a (mark))
        (indent-region a (mark))))))

(mapc #'fix
      (process-lines "git" "grep" "-l" "SPICEdev" "--" "*.c"))
