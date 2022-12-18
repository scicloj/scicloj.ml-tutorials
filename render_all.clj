(ns render-all
  (:require [notespace.cli :as cli]
            [notespace.api :as note]
            [clojure.java.shell]
            [nextjournal.clerk :as clerk]))

  
(def nss [
          {:ns 'scicloj.ml.intro                :output-file "docs/userguide-intro.html"}
          {:ns 'scicloj.ml.advanced             :output-file "docs/userguide-advanced.html"}
          {:ns 'scicloj.ml.categorical          :output-file "docs/userguide-categrical.html"}
          {:ns 'scicloj.ml.polyglot-kmeans      :output-file "docs/polyglot_kmeans.html"}
          {:ns 'scicloj.ml.transformers         :output-file "docs/userguide-transformers.html"}
          {:ns 'scicloj.ml.titanic              :output-file "docs/userguide-titanic.html"}
          {:ns 'scicloj.ml.tune-titanic         :output-file "docs/tune-titanic.html"}
          {:ns 'scicloj.ml.sklearnclj           :output-file "docs/userguide-sklearnclj.html"}
          {:ns 'scicloj.ml.third-party          :output-file "docs/userguide-third_party.html"}
          {:ns 'scicloj.ml.experiment-tracking  :output-file "docs/userguide-experiment-tracking.html"}
          {:ns 'scicloj.ml.unsupervised         :output-file "docs/userguide-unsupervised.html"}
          {:ns 'scicloj.ml.interactions-ols     :output-file "docs/interactions_ols.html"}])


(note/init :port 5678)

(run!

 #(do
    (println "render ns: " %)
    (cli/eval-and-render-a-notespace %))
 nss)

(require '[nextjournal.clerk :as clerk])

(clerk/build! {:paths ["src/scicloj/ml/models.clj"]
               :bundle? true
               :out-path "output"})

(println
 (clojure.java.shell/sh "mv" "output/index.html" "docs/userguide-models.html"))

(System/exit 0)
