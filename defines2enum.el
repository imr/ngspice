;;; (load buffer-file-name)
;;; (fix-all :commit)

;;-----------------------------------------------------------------------------

(defun split-list (list fun)
  "split the given list into sublists at every position
   where (fun prev-element next-element) is nil"
  (loop
     with accu
     with them
     for element in list
     do
       (progn
         (unless (and accu (funcall fun (first accu) element))
           (when accu
             (push (reverse accu) them))
           (setq accu nil))
         (push element accu))
     finally
       (progn
         (when accu
           (push (reverse accu) them))
         (return (reverse them)))))

(when nil
  (split-list '(1 2 3 7 8 9 11  13  15 16)
              (lambda (a b) (= (1+ a) b))))

(defun split-them (list-list fun)
  (loop
     for list in list-list
     append (split-list list fun)))

(when nil
  (split-them '((1 2 3 7 8 9 11  13  15 16) (23 24 26 27))
              (lambda (a b) (= (1+ a) b))))

;;-----------------------------------------------------------------------------

(defun collect-define-sequences ()
  "search sequences of [#define something digits]
   which are delminated by nothing except comments"
  (let ((them nil)
        (accu nil))
    (while (re-search-forward (rx bol
                                  (* space)
                                  (group "#define"
                                         (+ space) (group (+ (or alnum ?_)))
                                         (+ space) (group (+ digit)))
                                  (* (or space
                                         "\n"
                                         (seq "//" (* any) "\n")
                                         (seq "/*" (*?
                                                    (or
                                                     (not (any "*"))
                                                     (seq (+? "*") (not (any "/*")))))
                                              (+ "*") "/")))
                                  (group "#define"
                                         (+ space) (group (+ (or alnum ?_)))
                                         (+ space) (group (+ digit))))
                              nil t)
      (goto-char (match-beginning 4))
      (unless (and accu (equal (first (first accu))
                               (match-beginning 1)))
        (when accu
          (push (reverse accu) them)
          (setq accu nil))
        (push (list (match-beginning 1)
                    (match-string-no-properties 2)
                    (string-to-number (match-string-no-properties 3))
                    (match-end 3))
              accu))
      (push (list (match-beginning 4)
                  (match-string-no-properties 5)
                  (string-to-number (match-string-no-properties 6))
                  (match-end 6))
            accu))
    (when accu
      (push (reverse accu) them))
    (reverse them)))

;;-----------------------------------------------------------------------------

(defun predic (def1 def2)
  "whether def2 is the successor of def1"
  (= (1+ (third def1)) (third def2)))

;;-----------------------------------------------------------------------------

(when nil
  (let ((file "src/spicelib/devices/cap/capdefs.h"))
    (with-temp-buffer
      (insert-file-contents file)
      (split-them (collect-define-sequences) #'predic))))

;;; fixme consider preserve trailing comment
(defun fix (them)
  (loop
     ;; von hinten nach vorne, damit die positionen ...
     for list in (reverse them)
     for (pos1 name1 value1 end1) = (first list)
     for (pos2 name2 value2 end2) = (car (last list))
     when (>= (length list) 3)
     do (loop
           initially
             (progn
               (delete-region pos1 end2)
               (goto-char pos1)
               (when (looking-at (rx (+ (* space) "\n")))
                 (replace-match ""))
               (insert (format "enum {\n %s = %d,\n" name1 value1)))
           for (p1 name val) in (rest list)
           do
             (insert " " name ",\n")
           finally
             (progn
               (insert "};\n\n")
               (indent-region pos1 (point))))))

(defun fix-file (file)
  (with-temp-file file
    (insert-file-contents file)
    (c-mode)
    (setq c-file-style  "BSD")
    (setq c-basic-offset 4)
    (fix (split-them (collect-define-sequences) #'predic))))

(when nil
  (fix-file "src/spicelib/devices/cap/capdefs.h"))

(defun fix-all (&optional commit)
  (loop
   for file in (process-lines "git" "ls-files" "--" "*defs*.h")
   do (fix-file file))
  (when commit
    (process-lines "git" "commit" "-am" "execute defines2enum.el")))
