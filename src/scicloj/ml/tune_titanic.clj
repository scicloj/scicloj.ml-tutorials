(ns scicloj.ml.tune-titanic
  (:require
   [notespace.api :as note]
   [notespace.kinds :as kind]))
  
(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/tune-titanic.html")
  (note/init))


["This is the Clojure version of https://www.moritzkoerber.com/posts/preprocessing-hyperparameters/"]

(require  '[scicloj.ml.dataset :as ds]
          '[scicloj.ml.core :as ml]
          '[scicloj.ml.metamorph :as mm]
          '[camel-snake-kebab.core :as csk]
          '[scicloj.metamorph.ml.evaluation-handler :as eval-hn]
          '[tech.v3.datatype.functional :as dtfunc])

(def  categorical-features  [:pclass :sex :embarked])
(def  numeric-features [:age :parch :fare])

(defn map->vec [m] (flatten (into [] m)))

["Preproceesing Pipelines including feature engineering"]

(def data
  (-> (ds/dataset "data/titanic/train.csv"
                  {:key-fn csk/->kebab-case-keyword})
      (ds/select-columns (concat categorical-features numeric-features [:survived]))
      (ds/replace-missing categorical-features :value "missing")
      (ds/categorical->one-hot categorical-features)))


(defn replace-missing [options]
  (fn [ctx]
    (def ctx ctx)
    (def options options)
    ( (apply mm/replace-missing numeric-features (map->vec (:replace-missing-options options))) ctx)))

(defn maybe-std-scale [options]
  (fn [ctx]
    (def ctx ctx)
    (def options options)
    (if (-> options :scaling-options :scale?)
      ((mm/std-scale numeric-features {})
       ctx)
      ctx)))

(defn assoc-pipe-opts [options]
  (fn [ctx]
    (assoc ctx :pipe-options options)))


(defn make-decl-pipeline[model-type options]
  (def options options)
  [[::assoc-pipe-opts options]
   [::replace-missing options]
   [:mm/categorical->number [:survived ] {} :int64]
   [::maybe-std-scale options]
   [:mm/set-inference-target :survived]
   {:metamorph/id :model} [:mm/model (merge (:model-options options) {:model-type model-type})]])

(ml/->pipeline [[::replace-misssing options]])

(def logistic-regression-pipelines
  (map
   #(make-decl-pipeline :smile.classification/logistic-regression %)
   (ml/sobol-gridsearch {:scaling-options {:scale? (ml/categorical [true false])}
                         :replace-missing-options {:value (ml/categorical [dtfunc/mean dtfunc/median])}
                         :model-options {:lambda (ml/categorical [0.1 0.2 0.5 0.7 1])
                                         :tolerance (ml/categorical [0.1 0.01 0.001 0.0001])}})))

(def random-forrest-pipelines
  (map
   #(make-decl-pipeline :smile.classification/random-forest %)
   (ml/sobol-gridsearch {:scaling-options {:scale? (ml/categorical [true false])}
                         :replace-missing-options {:value (ml/categorical [dtfunc/mean dtfunc/median])}
                         :model-options {:trees (ml/categorical [5 50 100 250])
                                         :max_depth (ml/categorical [5 8 10])}})))

(def all-pipelines (concat random-forrest-pipelines))



(def pipe-fns
  (mapv ml/->pipeline all-pipelines))

["Simple split"]
(def splits (ds/split->seq data :holdout {:ratio 0.8}))
(def train-ds ((first splits) :train))
(def holdout-ds ((first splits) :test))

["Tune hyperparameter by evaluating all pipelines/models "]

(def files [atom []])
(def best-evaluation
  (ml/evaluate-pipelines
   all-pipelines
   (ds/split->seq train-ds :kfold 5)
   ml/classification-accuracy
   :accuracy
   {;; :evaluation-handler-fn (eval-hn/nippy-handler files "/tmp/titanic" "/home/carsten/Dropbox/sources/")
    :return-best-crossvalidation-only true
    :return-best-pipeline-only true}))

(def best-accuracy (-> best-evaluation first first :train-transform :metric))

["best accuracy found on train data: " best-accuracy]

(def best-options (-> best-evaluation first first :fit-ctx :pipe-options))
["best options found on train data: "]

best-options

(def best-pipe-fn
  (-> best-evaluation first first :pipe-fn))

best-pipe-fn

(def best-pipe-decl
  (-> best-evaluation first first :pipe-decl))

["best pipe"]
best-pipe-decl

(def source-refs
  (:fn-sources
   (scicloj.metamorph.ml.evaluation-handler/get-source-information
    [best-pipe-decl]
    *ns*
    (-> #'data meta :file))))

["pipe sources"]

(->>
 source-refs
 (filter #(let [v (val %)
                code-source (:code-source v)
                code-source-local (:code-local-source v)]
            (or code-source code-source-local)))
 (map (fn [[k v]]
        (def k k)
        (def v v)
        {k
         (str (:code-source v) (:code-local-source v))})))




 


(def predicted-survival-hold-out
  (->
   (best-pipe-fn
    (merge (-> best-evaluation first first :fit-ctx)
           {:metamorph/data holdout-ds :metamorph/mode :transform}))
   :metamorph/data
   ds/reverse-map-categorical-xforms
   :survived))

["Classication accuracy on holdout data: "]
(ml/classification-accuracy predicted-survival-hold-out
                           (holdout-ds :survived))

["Confusion matrix on holdout data"]
^kind/dataset
(->
 (ml/confusion-map predicted-survival-hold-out
                   (holdout-ds :survived))
 (ml/confusion-map->ds))

["Smile model object:"]
(ml/thaw-model
 (-> best-evaluation first first :fit-ctx :model))




["Feature importance:"]

(seq
 (.importance
  (ml/thaw-model
   (-> best-evaluation first first :fit-ctx :model))))



["## nested cross validation"]

(comment

  (require '[scicloj.ml.nested-cv :as nested-cv])


  (def nested-cv-result
    (doall
     (nested-cv/nested-cv data all-pipelines
                          ml/classification-accuracy
                          :accuracy 10 5)))



  ["nested cv best models metrics"]
  (map :metric nested-cv-result)

  (def final-model-by-cv
    (let [inner-k-fold (ds/split->seq data :kfold {:k 5})
          evaluation (ml/evaluate-pipelines
                      all-pipelines
                      inner-k-fold
                      ml/classification-accuracy
                      :accuracy)
          fit-ctx (-> evaluation first first :fit-ctx)
          best-pipefn (-> evaluation first first :pipe-fn)]
      {:best-pipe-fn best-pipefn
       :fit-ctx fit-ctx}))

  (def final-model
    ((:best-pipe-fn final-model-by-cv) {:metamorph/data data :metamorph/mode :fit}))

  ["Final best model"]
  (-> final-model
      (assoc-in [ :fit-ctx :model :model-data] nil)
      (assoc-in [:metamorph/data ] nil)
      (assoc-in [:model :model-data] nil))

  (def repeated
    (ds/split->seq data :kfold {:k 5 :repeats 5}))

  (-> (nth repeated 5) :train :age seq (#(take 5 %)))
  (-> (nth repeated 10) :train :age seq (#(take 5 %))))
