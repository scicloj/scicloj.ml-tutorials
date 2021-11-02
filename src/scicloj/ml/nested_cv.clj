(ns scicloj.ml.nested-cv
  (:require [tablecloth.api :as tc]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.classification :as clf]
            [tech.v3.datatype :as dt]))


(defn nested-cv [data pipelines metric-fn loss-or-accuracy outer-k inner-k]
  ;;  https://www.youtube.com/watch?v=DuDtXtKNpZs
  (let [k-folds (tc/split->seq data :kfold {:k outer-k})]
    (for [{train :train test :test} k-folds]
      (let [inner-k-fold (tc/split->seq test :kfold {:k inner-k})
            evaluation (ml/evaluate-pipelines
                        pipelines
                        inner-k-fold
                        metric-fn
                        loss-or-accuracy)
            fit-ctx (-> evaluation first first :fit-ctx)
            best-pipe-fn (-> evaluation first first :pipe-fn)
            transform-ctx (best-pipe-fn
                           (merge fit-ctx
                                  {:metamorph/data test :metamorph/mode :transform}))
            metric (metric-fn
                    (-> transform-ctx :model :scicloj.metamorph.ml/target-ds :survived dt/->vector)
                    (-> transform-ctx :metamorph/data :survived dt/->vector))]
        {:pipe-fn best-pipe-fn
         :fit-ctx fit-ctx
         :metric metric}))))
 
