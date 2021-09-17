(ns render-titanic
  (:require [notespace.cli :as cli]
            [notespace.api :as note]))

(note/init :port 5678)

(cli/eval-and-render-a-notespace {:ns 'scicloj.ml.titanic})
(System/exit 0)
