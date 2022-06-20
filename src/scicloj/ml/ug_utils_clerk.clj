(ns scicloj.ml.ug-utils-clerk

  (:require
   [nextjournal.clerk :as clerk]
   [clojure.string :as str]
   [notespace.kinds :as kind]
   [notespace.view :as view]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tablecloth.api :as tc]
   [libpython-clj2.python :as py]
   [tech.v3.datatype.functional :as dtf]
   [clj-http.client :as client]
   [scicloj.ml.ug-utils :as utils]))

(def model-key :smile.classification/discrete-naive-bayes)
(defn docu-options [model-key]

  (->
   (tc/dataset
    (or
     (get-in @scicloj.ml.core/model-definitions* [model-key :options])
     {:name [] :type [] :default []}))

   (tc/reorder-columns :name :type :default)))



(defn stringify-enum [form]
  (clojure.walk/postwalk (fn [x] (do (if  (instance? Enum x) (str x) x)))
                         form))

(defn render-key-info [prefix]
  (->> @scicloj.ml.core/model-definitions*
       (sort-by first)
       (filter #(str/starts-with? (first %) (str prefix)))
       (map
        (fn [[key definition]]
          (def key key)
          (def definition definition)
          [:div
           [:h3 {:id (str key)} (str key)]
           (utils/anchor-or-nothing (:javadoc (:documentation definition)) "javadoc")
           (utils/anchor-or-nothing (:user-guide (:documentation definition)) "user guide")

           ;; [:span (text->hiccup (or
           ;;                       (get-in @scicloj.ml.core/model-definitions* [key :documentation :description] ) ""))]

           [:span
            (let [docu-ds (docu-options key)]
              (if  (tc/empty-ds? docu-ds)
                nil
                (->
                 docu-ds
                 (tc/rows :as-maps)
                 seq
                 stringify-enum
                 (clerk/table))))]
            

           [:span
            (utils/docu-doc-string key)]

           [:hr]
           ;; [:div "Example:"]
           ;; [:div
           ;;  [:p/code {:code (str
           ;;                   (get-in definition [:documentation :code-example]
           ;;                           "" ))
           ;;            :bg-class "bg-light"}]]

           [:hr]]))))
