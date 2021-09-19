(ns render-all
  (:require [notespace.cli :as cli]
            [notespace.api :as note]))

  
(def nss [{:ns 'scicloj.ml.tune-titanic :output-file "docs/tune-titanic.html"}
          {:ns 'scicloj.ml.intro :output-file "docs/userguide-intro.html"}
          {:ns 'scicloj.ml.advanced :output-file "docs/userguide-advanced.html"}
          {:ns 'scicloj.ml.models :output-file "docs/userguide-models.html"}
          {:ns 'scicloj.ml.sklearnclj :output-file "docs/userguide-sklearnclj.html"}
          {:ns 'scicloj.ml.transformers :output-file "docs/userguide-transformers.html"}
          {:ns 'scicloj.ml.third-party :output-file "docs/userguide-third_party.html"}
          {:ns 'scicloj.ml.titanic :output-file "docs/userguide-titanic.html"}])


(note/init :port 5678)

(run!

 #(do
    (println "render ns: " %)
    (cli/eval-and-render-a-notespace %))
 nss)
 


;; (ns-publics 'scicloj.ml.intro)
(System/exit 0)
