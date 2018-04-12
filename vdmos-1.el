;;; (load-file buffer-file-name)

(defun mos1->vdmos(file)
    (with-temp-file file
      (insert-file-contents-literally file)
      (while (re-search-forward (rx "mos1")
                                nil t)
        (replace-match "vdmos"))))


(cl-loop for f in (process-lines "git" "ls-files" "--" "src/spicelib/devices/mos1")
         for dst = (concat "src/spicelib/devices/vdmos/"
                           (replace-regexp-in-string "mos1" "vdmos"
                                                     (file-name-nondirectory f)))
         initially (mkdir "src/spicelib/devices/vdmos" t)
         do (progn
              (copy-file f dst t)
              (mos1->vdmos dst)
              (process-lines "git" "add" dst)))
