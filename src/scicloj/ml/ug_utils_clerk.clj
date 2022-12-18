(ns scicloj.ml.ug-utils-clerk
  (:require
   [clojure.string :as str]
   [nextjournal.clerk :as clerk]
   [scicloj.ml.core :as ml]
   [scicloj.ml.ug-utils :as utils]
   [tablecloth.api :as tc]))

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
  (vec (concat [:span]
               (->> @scicloj.ml.core/model-definitions*
                    (sort-by first)
                    (filter #(str/starts-with? (first %) (str prefix)))
                    (mapv
                     (fn [[key definition]]
                       [:div
                        ;; (clerk/md (format "### %s" (str key)))
                        [:h3 {:id (str key)} (str key)]
                        (utils/anchor-or-nothing (:javadoc (:documentation definition)) "javadoc")
                        (utils/anchor-or-nothing (:user-guide (:documentation definition)) "user guide")

                        ;; [:span (text->hiccup (or
                        ;;                       (get-in @scicloj.ml.core/model-definitions* [key :documentation :description] ) ""))]

                        [:span

                         (let [docu-ds (docu-options key)]
                           (if  (tc/empty-ds? docu-ds)
                             ""
                             (->
                              docu-ds
                              (tc/rows :as-maps)
                              seq
                              stringify-enum
                              (clerk/table))))]
                        [:span
                         (utils/docu-doc-string key)]

                        [:hr]
                        [:hr]]))))))
