(ns render-titanic
  (:require [notespace.cli :as cli]))


(cli/eval-and-render-a-notespace {:ns 'scicloj.ml.titanic})
(System/exit 0)
