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
          '[tech.v3.datatype.functional :as dtfunc])

(def  categorical-features  [:pclass :sex :embarked])
(def  numeric-features [:age :parch :fare])

(defn map->vec [m] (flatten (into [] m)))


(def data
  (-> (ds/dataset "data/titanic/train.csv"
                  {:key-fn csk/->kebab-case-keyword})
      (ds/select-columns (concat categorical-features numeric-features [:survived]))
      (ds/replace-missing categorical-features :value "missing")
      (ds/categorical->one-hot categorical-features)))



(defn replace-missing [ds features k v]
  (def k k)
  (def v v)
  (mm/replace-missing features k v))


(defn make-pipeline-fns [model-type options]



  (ml/pipeline
   (fn [ctx]
     (assoc ctx :pipe-options options))
   (apply mm/replace-missing  numeric-features (map->vec (:replace-missing-options options)))
   (mm/categorical->number [:survived ] {} :int64)
   (ml/def-ctx ctx-1)
   (if (:scaling-options options)
     (mm/std-scale numeric-features {}))
   (mm/set-inference-target :survived)
   {:metamorph/id :model}
   (mm/model (merge (:model-options options)
                    {:model-type model-type}))))

(def logistic-regression-pipelines
  (map
   #(make-pipeline-fns :smile.classification/logistic-regression %)
   (ml/sobol-gridsearch {:scaling-options {:scale? (ml/categorical [true false])}
                         :replace-missing-options {:value (ml/categorical [dtfunc/mean dtfunc/median])}
                         :model-options {:lambda (ml/categorical [0.1 0.2 0.5 0.7 1])
                                         :tolerance (ml/categorical [0.1 0.01 0.001 0.0001])}})))

(def random-forrest-pipelines
  (map
   #(make-pipeline-fns :smile.classification/random-forest %)
   (ml/sobol-gridsearch {:scaling-options {:scale? (ml/categorical [true false])}
                         :replace-missing-options {:value (ml/categorical [dtfunc/mean dtfunc/median])}
                         :model-options {:trees (ml/categorical [5 50 100 250])
                                         :max_depth (ml/categorical [5 8 10])}})))

(def all-pipelines (concat random-forrest-pipelines logistic-regression-pipelines))


(def splits (ds/split->seq data :holdout {:ratio 0.8}))
(def train-ds ((first splits) :train))
(def holdout-ds ((first splits) :test))


(def best-evaluation
  (ml/evaluate-pipelines
   all-pipelines
   (ds/split->seq train-ds :kfold 5)
   ml/classification-accuracy :accuracy
   {:return-best-crossvalidation-only true
    :return-best-pipeline-only true}))

(def best-accuracy (-> best-evaluation first first :metric))

["best test accuracy: " best-accuracy]

(def best-options (-> best-evaluation first first :fit-ctx :pipe-options))
["best test options: " best-options]


(def best-pipe-fn
  (-> best-evaluation first first :pipe-fn))

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
