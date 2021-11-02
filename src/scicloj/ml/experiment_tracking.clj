(ns scicloj.ml.experiment-tracking
  (:require
     [notespace.api :as note]
     [notespace.kinds :as kind]))

(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/userguide-experiment-tracking.html")
  (note/init))

(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset  :as ds]
         '[scicloj.metamorph.ml.tools :refer [dissoc-in]]
         '[taoensso.nippy :as nippy])


(defonce ds (ds/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

(defn create-base-pipe-decl [node-size]
  [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
   [:tech.v3.dataset.metamorph/categorical->number [:species]]
   {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest
                                                        :node-size node-size}]])
["## Run evaluation"]

["We create 6 pipelines,  do a simple :holdout split and keep all results. In order to save memory,
as we needed to do, if we would have thousands or more evaluations, we keep the minimal information."]

(def pipes (map create-base-pipe-decl [1 5 10 20 50 100]))
(def split (ds/split->seq ds :holdout))

(def  evaluation-result
  (ml/evaluate-pipelines
   pipes split
   ml/classification-accuracy
   :accuracy
   {:result-dissoc-in-seq ml/result-dissoc-in-seq--all
    :return-best-crossvalidation-only false
    :return-best-pipeline-only false}))

["So we get here 6 evaluation results"]
(->> evaluation-result flatten
     (map (comp :metric :test-transform)))

["## Attach a simple result handler"]

["A result handler is a function which takes a full map representing a single evalution result and does what ever is needed.
It is by default a function with side effects, and it should return nil."]

["The function will be called for each evalution result, so in our case 6 times. We use a simple function for now,
which prints the current declartive pipeline."]

(def  evaluation-result
  (ml/evaluate-pipelines
   pipes split
   ml/classification-accuracy
   :accuracy
   {:result-dissoc-in-seq ml/result-dissoc-in-seq--all
    :return-best-crossvalidation-only false
    :return-best-pipeline-only false
    :evaluation-handler-fn
    (fn [result]
      (clojure.pprint/pprint (:pipe-decl result)))}))

["repl output: "]
^kind/code
[[:tech.v3.dataset.metamorph/set-inference-target [:species]]
 [:tech.v3.dataset.metamorph/categorical->number [:species]]
 [:scicloj.metamorph.ml/model
  {:model-type :smile.classification/random-forest, :node-size 1}]]
[[:tech.v3.dataset.metamorph/set-inference-target [:species]]
 [:tech.v3.dataset.metamorph/categorical->number [:species]]
 [:scicloj.metamorph.ml/model
  {:model-type :smile.classification/random-forest, :node-size 5}]]
[[:tech.v3.dataset.metamorph/set-inference-target [:species]]
 [:tech.v3.dataset.metamorph/categorical->number [:species]]
 [:scicloj.metamorph.ml/model
  {:model-type :smile.classification/random-forest, :node-size 10}]]
[[:tech.v3.dataset.metamorph/set-inference-target [:species]]
 [:tech.v3.dataset.metamorph/categorical->number [:species]]
 [:scicloj.metamorph.ml/model
  {:model-type :smile.classification/random-forest, :node-size 20}]]
["...."]

["The callback function can now implement whatever needed to store teh evaluation results, for example on disk.
"]


["Write results to disk"]



(def  evaluation-result
  (ml/evaluate-pipelines
   pipes split
   ml/classification-accuracy
   :accuracy
   {:evaluation-handler-fn
    (scicloj.metamorph.ml.evaluation-handler/example-nippy-handler created-files "/tmp" [[:fit-ctx :model :model-data]
                                                                                         [:train-transform :ctx :model :model-data]
                                                                                         [:test-transform :ctx :model :model-data]])

    :attach-fn-sources {:ns (find-ns 'scicloj.ml.experiment-tracking)
                        :pipe-fns-clj-file "src/scicloj/ml/experiment_tracking.clj"}}))

["This creates one nippy file for each evaluation, containing all data of the evaluations."]
