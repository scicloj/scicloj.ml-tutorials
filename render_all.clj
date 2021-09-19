(ns render-all
  (:require [notespace.cli :as cli]
            [notespace.api :as note]))

  
(def nss ['scicloj.ml.intro
          'scicloj.ml.advanced
          'scicloj.ml.models
          'scicloj.ml.sklearnclj
          'scicloj.ml.transformers
          'scicloj.ml.third-party
          'scicloj.ml.titanic
          'scucloj.ml.tune-titanic])

(note/init :port 5678)

(run!
 #(cli/eval-and-render-a-notespace {:ns %})
 nss)
 


;; (ns-publics 'scicloj.ml.intro)
(System/exit 0)
