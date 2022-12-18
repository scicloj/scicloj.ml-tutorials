(ns scicloj.ml.categorical

  (:require [notespace.api :as note]
            [notespace.kinds :as kind]))


(comment
  (note/init-with-browser)
  (note/eval-and-realize-this-notespace))

(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset  :as ds])

["# Handling of categorical variables"]

["We keep important information in the metadata of the column,
which can be inspected"]

["## categorical -> number"]
["Categorical columns can be converted too numbers, which is needed by several ML models."]

(def ds-cat
  (ds/dataset {:a [:x :y :x]}))

["inspect column metadata and observe datatype :kewyword"]
(-> ds-cat :a meta)

["convert categorical columns to numeric"]
(def ds-number
  (ds/categorical->number
   ds-cat :all {} :int))

^kind/dataset
ds-number

["metadata has changed as well, int now, and with a lookup table"]
(-> ds-number :a meta)




["## categorical -> one-hot"]
["Categorical columns can be converted to one-hot columns as well, which is needed by several ML models."]
(def ds-one-hot
  (ds/categorical->one-hot
   ds-cat :all {} :int))

^kind/dataset
ds-one-hot


["we can go back as well"]
(-> ds-one-hot ds/reverse-map-categorical-xforms)


["inspect metadata after conversion"]
(-> ds-one-hot :a-y meta)


["we can go back"]
(-> ds-one-hot ds/reverse-map-categorical-xforms)
