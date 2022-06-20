(ns scicloj.ml.ug-utils
  (:require [clojure.string :as str]
            [notespace.kinds :as kind]
            [notespace.view :as view]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tablecloth.api :as tc]
            [libpython-clj2.python :as py]
            [tech.v3.datatype.functional :as dtf]
            [clj-http.client :as client]))
            

(defn kroki [s type format]
  (client/post "https://kroki.io/" {:content-type :json
                                    :as :byte-array
                                    :form-params
                                    {:diagram_source s
                                     :diagram_type (name type)
                                     :output_format (name format)}}))
(py/initialize!)
(def doc->markdown (py/import-module "docstring_to_markdown"))



(def model-keys
  (keys @scicloj.ml.core/model-definitions*))

(def model-options
  (map
   :options
   (vals @scicloj.ml.core/model-definitions*)))

(defn dataset->md-hiccup [mds]
  (let [height (* 46 (- (count (str/split-lines (str mds))) 2))
        height-limit (min height 800)]
    [:div {:class "table table-striped table-hover table-condensed table-responsive"}
           ;; :style {:height (str height-limit "px")}
           
     (view/markdowns->hiccup mds)]))


(defmethod kind/kind->behaviour ::dataset-nocode
  [_]
  {:render-src?   false
   :value->hiccup #'dataset->md-hiccup})

(defn docu-options [model-key]
  (kind/override
   (->
    (tc/dataset
     (or
      (get-in @scicloj.ml.core/model-definitions* [model-key :options])
      {:name [] :type [] :default []}))

    (tc/reorder-columns :name :type :default))

   ::dataset-nocode))
   
  


;; (->
;;  (tc/dataset
;;   (get-in @scicloj.ml.core/model-definitions* [:corenlp/crf :options] ))
;; (tc/reorder-columns :name :type :default)
;;  )

(defn text->hiccup
  "Convert newlines to [:br]'s."
  [text]
  (->> (str/split text #"\n")
       (interpose [:br])
       (map #(if (string? %)
               %
               (with-meta % {:key (gensym "br-")})))))

(defn docu-doc-string [model-key]
  (try
    (view/markdowns->hiccup
     (py/py. doc->markdown convert
             (or
              (get-in @scicloj.ml.core/model-definitions* [model-key :documentation :doc-string] ) "")))
    (catch Exception e "")))




(defn anchor-or-nothing [x text]
  (if (empty? x)
    [:div ""]
    [:div
     [:a {:href x} text]]))
    
  

(defn render-key-info [prefix]
  (->> @scicloj.ml.core/model-definitions*
       (sort-by first)
       (filter #(str/starts-with? (first %) (str prefix)))
       (map
        (fn [[key definition]]
          [:div
           [:h3 {:id (str key)} (str key)]
           (anchor-or-nothing (:javadoc (:documentation definition)) "javadoc")
           (anchor-or-nothing (:user-guide (:documentation definition)) "user guide")

           ;; [:span (text->hiccup (or
           ;;                       (get-in @scicloj.ml.core/model-definitions* [key :documentation :description] ) ""))]

           [:span
            (dataset->md-hiccup (docu-options key))]

           [:span
            (docu-doc-string key)]

           [:hr]
           ;; [:div "Example:"]
           ;; [:div
           ;;  [:p/code {:code (str
           ;;                   (get-in definition [:documentation :code-example]
           ;;                           "" ))
           ;;            :bg-class "bg-light"}]]

           [:hr]]))))
           

(text->hiccup (or
               (get-in @scicloj.ml.core/model-definitions*
                       [:smile.manifold/tsne :documentation :description]) ""))


(defn remove-deep [key-set data]
  (clojure.walk/prewalk (fn [node] (if (map? node)
                                    (apply dissoc node key-set)
                                    node))
                        data))
(defn stepped-range [start end n-steps]
  (let [diff (- end start)]
    (range start end (/ diff n-steps))))

(defn surface-plot [iris cols raw-pipe-fn model-name]
  (let [
        pipe-fn
        (ml/pipeline
         (mm/select-columns (concat [:species] cols))
         raw-pipe-fn)



        fitted-ctx
        (pipe-fn
         {:metamorph/data iris
          :metamorph/mode :fit})


        _ (def fitted-ctx fitted-ctx)
        ;; getting plot boundaries
        min-x (- (-> (get iris (first cols)) dtf/reduce-min) 0.2)
        min-y (- (-> (get iris (second cols)) dtf/reduce-min) 0.2)
        max-x (+ (-> (get iris (first cols)) dtf/reduce-max) 0.2)
        max-y (+ (-> (get iris (second cols)) dtf/reduce-max) 0.2)


        ;; make a grid for the decision surface
        grid
        (for [x1 (stepped-range min-x max-x 100)
              x2 (stepped-range min-y max-y 100)]

          {(first cols) x1
           (second cols) x2
           :species nil})

        grid-ds (tc/dataset grid)


        _ (def grid-ds grid-ds)
        _ (def fitted-ctx fitted-ctx)
        _ (def pipe-fn pipe-fn)
        ;; predict for all grid points
        prediction-grid
        (->
         (pipe-fn
          (merge
           fitted-ctx
           {:metamorph/data grid-ds
            :metamorph/mode :transform}))
         :metamorph/data
         (ds-mod/column-values->categorical :species)
         seq)

        ;; (def x
        ;;   (->
        ;;    (pipe-fn
        ;;     (merge
        ;;      fitted-ctx
        ;;      {:metamorph/data grid-ds
        ;;       :metamorph/mode :transform}))
        ;;    :metamorph/data))



        ;; (ds-mod/dataset->categorical-xforms x)
        ;; (ds-mod/column-values->categorical x :species)
        ;; (tech.v3.dataset.categorical/dataset->categorical-maps x)




        grid-ds-prediction
        (tc/add-column grid-ds :predicted-species prediction-grid)


        ;; predict the iris data points from data set
        prediction-iris
        (->
         (pipe-fn
          (merge
           fitted-ctx
           {:metamorph/data iris
            :metamorph/mode :transform}))
         :metamorph/data

         (ds-mod/column-values->categorical :species)
         seq)

        ds-prediction
        (tc/add-column iris :true-species (:species iris)
                       prediction-iris)]

    ;; create a 2 layer Vega lite specification
    {:layer
     [

      {:data {:values (seq (tc/rows grid-ds-prediction :as-maps))}
       :title (str "Decision surfaces for model: " model-name)
       :width 500
       :height 500
       :mark {:type "square" :opacity 0.9 :strokeOpacity 0.1 :stroke nil},
       :encoding {:x {:field (first cols)
                      :type "quantitative"
                      :scale {:domain [min-x max-x]}
                      :axis {:format "2.2"
                             :labelOverlap true}}
                      
                  :y {:field (second cols) :type "quantitative"
                      :axis {:format "2.2"
                             :labelOverlap true}
                      :scale {:domain [min-y max-y]}}
                      
                  :color {:field :predicted-species}}}
                  

      {:data {:values (seq (tc/rows ds-prediction :as-maps))}

       :width 500
       :height 500
       :mark {:type "circle" :opacity 1 :strokeOpacity 1},
       :encoding {:x {:field (first cols)
                      :type "quantitative"
                      :axis {:format "2.2"
                             :labelOverlap true}
                      :scale {:domain [min-x max-x]}}
                      
                  :y {:field (second cols) :type "quantitative"
                      :axis {:format "2.2"
                             :labelOverlap true}
                      :scale {:domain [min-y max-y]}}
                      

                  :fill {:field :true-species} ;; :legend nil
                         
                  :stroke { :value :black}
                  :size {:value 300}}}]}))

(defn select-paths-from-set [current-path path-set data]
  (cond
    (map? data) (into {}
                      (remove nil?)
                      (for [[k v] data]
                        (let [p (conj current-path k)]
                          (if (contains? path-set p)
                            [k (select-paths-from-set p path-set v)]))))
    (sequential? data) (mapv (partial select-paths-from-set current-path path-set) data)
    :default data))

(defn select-paths [data paths]
  (select-paths-from-set []
                         (into #{}
                               (mapcat #(take-while seq (iterate butlast %)))
                               paths)
                         data))

(defn select-minimal-result [result]
    (select-paths result [[:train-transform :metric]
                          [:test-transform :metric]]))
