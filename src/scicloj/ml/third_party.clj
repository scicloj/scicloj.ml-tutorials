(ns scicloj.ml.third-party
 (:require [notespace.api :as note]
           [notespace.kinds :as kind]
           [scicloj.ml.ug-utils :refer :all]
           [dk.simongray.datalinguist.ml.crf]
           [clj-djl.mmml]))

(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/userguide-third_party.html")
  (note/init))
  



(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset  :as ds]
         '[tech.v3.datatype.functional :as dfn]
         '[clojure.tools.namespace.find :as ns-find]
         '[clojure.java.classpath :as cp]
         '[scicloj.ml.xgboost]
         '[camel-snake-kebab.core :as csk])

(comment
  (->> (cp/classpath)
       (ns-find/find-ns-decls)
       (map second)
       (filter #(clojure.string/includes? (name %) "kebab"))))




["# xgboost"]
["## Example code"]

(def house-price
  (->
   (ds/dataset
    "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv" {:key-fn csk/->kebab-case-keyword})
   (ds/replace-missing :type/string "NA")
   (ds/categorical->number  #(ds/select-columns % :type/string))))


(def split (first (ds/split->seq house-price :holdout)))

(def train-ds (:train split))
(def test-ds (:test split))


(def pipe-fn
  (ml/pipeline
   (mm/replace-missing :type/numerical :value 0)
   (mm/set-inference-target :sale-price)
   {:metamorph/id :model} (mm/model {:model-type :xgboost/linear-regression})))

(def fit-result
  (let [fitted-ctx
        (ml/fit-pipe train-ds pipe-fn)
        test-predictions
        (ml/transform-pipe test-ds pipe-fn fitted-ctx)
        error
        (ml/mae (-> test-predictions  :metamorph/data :sale-price)
                (-> test-ds :sale-price))]
    {:error error
     :gains (->
             (ml/explain (-> fitted-ctx :model))
             (ds/order-by :gain :desc))}))



["error:"]
(:error fit-result)

["Feature importance - gain"]

^kind/dataset
(:gains fit-result)

["## Reference"]

^kind/hiccup-nocode (render-key-info ":xgboost")

["# Deep learning models via clj-djl "]



(def train-ds
  (ds/dataset
   "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"))


(def test-ds
  (->
   (ds/dataset
    "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv")
   (ds/add-column "SalePrice" 0)))

(defn numeric-features [ds]
  (ds/intersection (ds/numeric ds)
                   (ds/feature ds)))

(defn update-columns
  "Update a sequence of columns selected by column name seq or column selector function."
  [dataframe col-name-seq-or-fn update-fn]
  (ds/update-columns dataframe
                     (if (fn? col-name-seq-or-fn)
                       (ds/column-names (col-name-seq-or-fn dataframe))
                       col-name-seq-or-fn)
                     update-fn))




(require
 '[clj-djl.nn :as nn]
 '[clj-djl.training :as t]
 '[clj-djl.training.loss :as loss]
 '[clj-djl.training.optimizer :as optimizer]
 '[clj-djl.training.tracker :as tracker]
 '[clj-djl.training.listener :as listener]
 '[clj-djl.ndarray :as nd]
 '[clj-djl.nn.parameter :as param])

(def  learning-rate 0.05)
(defn net [] (nn/sequential {:blocks (nn/linear {:units 1})
                             :initializer (nn/normal-initializer)
                             :parameter param/weight}))

(defn cfg [] (t/training-config {:loss (loss/l2-loss)
                                 :optimizer (optimizer/sgd
                                             {:tracker (tracker/fixed learning-rate)})
                                 :evaluator (t/accuracy)
                                 :listeners (listener/logging)}))



(def pipe
  (ml/pipeline

   (mm/drop-columns ["Id"])
   (mm/set-inference-target "SalePrice")
   (mm/replace-missing :type/numerical :value 0)
   (mm/replace-missing :!type/numerical :value "None")
   (ml/lift update-columns numeric-features
            #(dfn// (dfn/- % (dfn/mean %))
                    (dfn/standard-deviation %)))
   (mm/transform-one-hot :!type/numerical :full)
   (mm/update-column "SalePrice"
                     #(dfn// % (dfn/mean %)))

   (mm/set-inference-target "SalePrice")

   (mm/model {:model-type :clj-djl/djl
              :batchsize 64
              :model-spec {:name "mlp" :block-fn net}
              :model-cfg (cfg)
              :initial-shape (nd/shape 1 311)
              :nepoch 1})))




(def trained-pipeline
  (pipe {:metamorph/data train-ds
         :metamorph/mode :fit
         :metamorph.ml/full-ds (ds/concat train-ds test-ds)}))


         
(def predicted-pipeline
  (pipe
   (merge trained-pipeline
          {:metamorph/data test-ds
           :metamorph/mode :transform})))




( get
 (:metamorph/data predicted-pipeline)
 "SalePrice")


^kind/hiccup-nocode
(render-key-info ":clj-djl/djl")


["# A NER model from Standford CorenLP"]

^kind/hiccup-nocode
(render-key-info ":corenlp")

