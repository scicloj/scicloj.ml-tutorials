(ns scicloj.ml.interactions-ols
  (:require
   [notespace.api :as note]
   [notespace.kinds :as kind]
   [notespace.view :as view]
   [tablecloth.api :as tc]
   [scicloj.ml.core]
   [scicloj.sklearn-clj.ml]
   [clojure.string :as str]
   [scicloj.ml.ug-utils :refer :all]
   [clojure.java.io :as io]
   [fastmath.stats :as fmstats]))

(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset :refer [dataset add-column]]
         '[scicloj.ml.dataset :as ds]
         '[tech.v3.dataset.math :as std-math]
         '[tech.v3.datatype.functional :as dtf]
         '[scicloj.metamorph.ml.toydata :as datasets])


(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/render-static-html "docs/interactions_ols.html"))

["This examples how, how to do interactions in linear regression with `scicloj.ml`"]

["Taking ideas from: "

 "http://www.sthda.com/english/articles/40-regression-analysis/164-interaction-effect-in-multiple-regression-essentials/#comments-list"]

(defn pp-str [x]
  (with-out-str (clojure.pprint/pprint x)))

["First we load the data:"]
(def marketing (tc/dataset "data/marketing.csv" {:key-fn keyword}))

["## Additive model"]

["Firts we build an additive model, which model equation is 'sales = b0 + b1 * youtube + b2 * facebook'"]

(def additive-pipeline
  (ml/pipeline
   (mm/set-inference-target :sales)
   (mm/drop-columns [:newspaper])
   {:metamorph/id :model}
   (mm/model {:model-type :smile.regression/ordinary-least-square})))


["We evaluate it, "]
(def evaluations
  (ml/evaluate-pipelines
   [additive-pipeline]
   (ds/split->seq marketing :holdout)
   ml/rmse
   :loss
   {:other-metrices [{:name :r2
                      :metric-fn fmstats/r2-determination}]}))


["and print the result:"]
^kind/hiccup
(text->hiccup
 (str
  (-> evaluations flatten first :fit-ctx :model ml/thaw-model str)))

["We have the following metrices:"]
["RMSE"]
(-> evaluations flatten first :test-transform :metric)

["R2"]
(-> evaluations flatten first :test-transform :other-metrices first :metric)

["## Interaction effects"]
["Now we add interaction effects to it, resulting in this model equation: 'sales = b0 + b1 * youtube + b2 * facebook + b3 * (youtube * facebook)'"]
(def pipe-interaction
  (ml/pipeline
   (mm/drop-columns [:newspaper])
   (mm/add-column :youtube*facebook (fn [ds] (dtf/* (ds :youtube) (ds :facebook))))
   (mm/set-inference-target :sales)
   {:metamorph/id :model}(mm/model {:model-type :smile.regression/ordinary-least-square})))

["Again we evaluate the model,"]
(def evaluations
  (ml/evaluate-pipelines
   [pipe-interaction]
   (ds/split->seq marketing :holdout)
   ml/rmse
   :loss
   {:other-metrices [{:name :r2
                      :metric-fn fmstats/r2-determination}]}))


["and print it and the performance metrices:"]
^kind/hiccup
(text->hiccup
 (str
  (-> evaluations flatten first :fit-ctx :model ml/thaw-model str)))

["As the multiplcation of 'youtube * facebook' is as well statistically relevant, it
suggests that there is indeed an interaction between these 2 predictor variables youtube and facebook."]

["RMSE"]
(-> evaluations flatten first :test-transform :metric)

["R2"]
(-> evaluations flatten first :test-transform :other-metrices first :metric)

["RMSE and R2 of the intercation model are sligtly better."
 "These results suggest that the model with the interaction term is better than the model that contains only main effects.
So, for this specific data, we should go for the model with the interaction model.
"]
