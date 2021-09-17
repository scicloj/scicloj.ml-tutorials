(ns render-all
  (:require [notespace.cli :as cli]))

  
(def nss ['scicloj.ml.intro
          'scicloj.ml.advanced
          'scicloj.ml.models
          'scicloj.ml.sklearnclj
          'scicloj.ml.transformers
          'scicloj.ml.third-party
          'scicloj.ml.titanic])


(run!
 #(cli/eval-and-render-a-notespace {:ns %})
 nss)
 


;; (ns-publics 'scicloj.ml.intro)
(System/exit 0)
