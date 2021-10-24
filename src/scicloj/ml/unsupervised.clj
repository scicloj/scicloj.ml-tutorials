(ns scicloj.ml.unsupervised
  (:require
   [notespace.api :as note]
   [notespace.kinds :as kind]
   [net.clojars.behrica.cluster_eval :as cluster-eval]))



(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/userguide-unsupervised.html")
  (note/init))

(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset  :as ds])

["# Iris data"]

(def iris
  (->
   (ds/dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))




["# k-means clustering"]

(def fit-ctx
  (ml/fit
   iris
   (mm/select-columns [:petal_length :petal_width])
   {:metamorph/id :model}
   (mm/model {:model-type :fastmath/cluster
              :clustering-method :k-means
              :clustering-method-args [3]})))

(def iris-with-cluster
  (ds/add-column iris :cluster
                 (-> fit-ctx :model :model-data :clustering)))

(def centroids
  (map
   (fn [[petal-length petal-width]]
     (hash-map :petal_length petal-length
               :petal_width petal-width))
   (-> fit-ctx :model :model-data :representatives)))

^kind/vega
{:height 300
 :width 300

 :title "2D result of iris k-means clustering with cluster centroids (n=3)"
 :layer [{
          :$schema "https://vega.github.io/schema/vega-lite/v5.json"
          :data {:values (ds/rows iris-with-cluster :as-maps)}
          :description "Iris data "
          :encoding {:x {:field :petal_length :type "quantitative"}
                     :y {:field :petal_width :type "quantitative"}
                     :color {:field :cluster}}
          :mark "point"}
         {
          :data {:values centroids}
          :description "Iris data "
          :encoding {:x {:field :petal_length :type "quantitative"}
                     :y {:field :petal_width :type "quantitative"}}

          :mark {:type "point" :shape :triangle-up :color :black
                 :filled true
                 :size 200}}]}

         

["# Ellbow plot"]

["## Calculate distortion over n"]

(defn make-pipe [n]
  (ml/pipeline
   (mm/drop-columns [:species])
   {:metamorph/id :model}
   (mm/model {:model-type :fastmath/cluster
              :clustering-method :k-means
              :clustering-method-args [n]})))



(def eval-results
  (ml/evaluate-pipelines
   (map make-pipe (range 2 10))
   [{:train iris}]
   (fn [ctx]
     0)
   :loss
   {:return-best-pipeline-only false}))



(defn fastmath->cluster-data [model-data]
  (def model-data model-data)
  (let [
        cluster-values
        (concat
         (-> model-data :data)
         (-> model-data :representatives))

        cluster
        (concat
         (-> model-data :clustering)
         (range (-> model-data :representatives count)))

        centroid?
        (concat
         (repeat (-> model-data :data count) false)
         (repeat (-> model-data :representatives count) true))]

    {:values cluster-values
     :cluster cluster
     :centroid? centroid?}))



(def ellbow-plot-data-distortion
  (map #(hash-map :n %1
                  :distortion %2)
       (->> eval-results flatten (map #(first (get-in % [:fit-ctx :model :options :clustering-method-args]))))
       (->> eval-results flatten (map #(get-in % [:fit-ctx :model :model-data :info :distortion])))))
        

["## Calculate silouhette score over n"]

(def eval-results-silhouete
  (ml/evaluate-pipelines
   (map make-pipe (range 2 10))
   [{:train iris}]
   (fn [ctx]
     (let [metric
           (cluster-eval/cluster-index
            (fastmath->cluster-data (-> ctx :model :model-data))
            "calcularSilhouette")]
       metric))
   :loss
   {:return-best-pipeline-only false}))


(def ellbow-plot-data-silhoute
  (map #(hash-map :n %1
                  :silhoute %2)
       (->> eval-results-silhouete flatten (map #(first (get-in % [:fit-ctx :model :options :clustering-method-args]))))
       (->> eval-results-silhouete flatten (map #(get-in % [:train-transform :metric])))))


["Ellbow plots for distortion and silhoute score"]

^kind/vega
{:hconcat [
           {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
            :width 200
            :height 200
            :title "Ellbow plot of distortion for various n"
            :data {:values ellbow-plot-data-distortion}
            :description "Stock prices of 5 Tech Companies over Time."
            :encoding {:x {:field "n" :type :ordinal}
                       :y {:field :distortion :type "quantitative"}}
            :mark {:point true :type "line"}}

           {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
            :width 200
            :height 200
            :title "Ellbow plot of Silhoutte score for various n"
            :data {:values ellbow-plot-data-silhoute}

            :encoding {:x {:field "n" :type :ordinal}
                       :y {:field :silhoute :type "quantitative"}}
            :mark {:point true :type "line"}}]}
