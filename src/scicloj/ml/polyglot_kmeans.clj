(ns scicloj.ml.polyglot-kmeans
  (:require
   [scicloj.sklearn-clj.metamorph]
   [nextjournal.clerk :as clerk]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :as py :refer [py.- py.]]))

(comment
  (clerk/serve! {:browser true})
  (clerk/build-static-app! {:paths ["src/scicloj/ml/polyglot_kmeans.clj"]
                            :bundle? false})
  (clerk/clear-cache!))

^{::clerk/visibility #{:hide}}
(clerk/code
 "
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    init=\"random\",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42)

kmeans.fit(scaled_features)

kmeans.inertia_
")



;; # 1. Use libpython-clj
;; This is using the same python classes as above
;; So it is "the same code"
;;
(require-python '[sklearn.datasets :refer [make_blobs]]
                '[sklearn.preprocessing :refer [StandardScaler]]
                '[sklearn.cluster :refer [KMeans]])



(def blobs
  (make_blobs :n_samples 200
              :n_features 50
              :centers 3
              :cluster_std 2.75
              :random_state 42))

(def scaler (StandardScaler))
(def features (first blobs))
(def scaled-features (py. scaler fit_transform features))
(def k-means (KMeans
              :init "random"
              :n_clusters 3
              :n_init 10
              :max_iter 300
              :random_state 42))
(py. k-means fit scaled-features)
(py.- k-means inertia_)

(println :python
 (py.- k-means inertia_))


;; # 2. use sklearn-clj
;; This librraies allow to use all estimators/model from sklearn
;; It uses libpython-clj, but "hidden" behind sklearn-clj
;;

(require '[scicloj.ml.sklearnclj])
(require '[scicloj.ml.dataset :as ds]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.core :as ml]
         '[scicloj.sklearn-clj.metamorph :as sklearn-clj])


(def data (-> blobs first py/->jvm ds/dataset))

(def fitted-ctx-1
  (ml/fit
   data
   (mm/std-scale  :all {})
   {:metamorph/id :k-means}
   (sklearn-clj/estimate
    :sklearn.cluster "KMeans"
    {:init "random"
     :n_clusters 3
     :n_init 10
     :max_iter 300
     :random_state 42})))
(-> fitted-ctx-1 :k-means :attributes :inertia_)

(println :sklearnclj
 (-> fitted-ctx-1 :k-means :attributes :inertia_))

;; # 3. use Clojure only pipeline
;;  So no python interop in use
;;  It uses clustering algorithms from JVM library Smile

(require '[scicloj.ml.smile.clustering :as clustering])

(def fitted-ctx-2
  (ml/fit
   data
   (mm/std-scale  :all {})
   {:metamorph/id :k-means}
   (scicloj.ml.smile.clustering/cluster
    :k-means
    [3 300]
    :cluster)))

(-> fitted-ctx-2 :k-means  :info :distortion)

(println :scicloj-ml
         (-> fitted-ctx-2 :k-means :info :distortion))

;; # 4. use declarative Clojure only pipeline
;; same as 3), only using metamorph declarative pipelines



(def decl-pipe
  [[:mm/std-scale :all {}]
   {:metamorph/id :k-means}
   [:scicloj.ml.smile.clustering/cluster
    :k-means
    [3 300]
    :cluster]])

(def distortion-1
  (->> decl-pipe
       ml/->pipeline
       (ml/fit-pipe data)
       :k-means
       :info
       :distortion))

(println :scicloj-ml-decl
         distortion-1)

(frequencies
 (repeatedly 1000 (fn []
                    (->> decl-pipe
                         ml/->pipeline
                         (ml/fit-pipe data)
                         :k-means
                         :info
                         :distortion))));; => {6704.217542681282 1,
;;     6123.509172624183 1,
;;     6111.305104765819 1,
;;     6123.512790872804 1,
;;     6128.022263811994 1,
;;     6127.5249232455435 1,
;;     6112.912230308002 1,
;;     6627.690117803246 1,
;;     6640.623676535145 1,
;;     6138.478847922785 1,
;;     6117.197438733556 1,
;;     6126.689844941797 1,
;;     6641.747215501464 1,
;;     6121.237719400343 1,
;;     6117.973100694475 1,
;;     6159.275009071018 1,
;;     6128.118378321051 1,
;;     6155.31465901093 1,
;;     6113.229384413742 1,
;;     6119.921270620962 1,
;;     6640.885024710563 1,
;;     6122.960751607536 1,
;;     6144.273578776018 1,
;;     6118.653479575775 1,
;;     6118.798971999091 1,
;;     6115.158116221894 1,
;;     6115.995302285262 1,
;;     6117.809538867476 1,
;;     6124.8556874039805 1,
;;     6622.24593020113 1,
;;     6116.495933646238 1,
;;     6119.422503524379 1,
;;     6115.604384221734 1,
;;     6119.97738537433 1,
;;     6125.311210839226 1,
;;     6675.159239310862 1,
;;     6176.945686021643 3,
;;     6151.346352698883 1,
;;     6120.209608958619 1,
;;     6123.933983888712 1,
;;     6128.329747242062 1,
;;     6157.47713671024 1,
;;     6671.016523153647 1,
;;     3113.1273632925795 913,
;;     6119.532974408946 1,
;;     6682.724995352249 1,
;;     6121.438813953806 1,
;;     6657.719107153658 1,
;;     6121.982856827681 1,
;;     6656.54703296517 1,
;;     6130.360501698988 1,
;;     6137.228874482338 1,
;;     6671.453521188791 1,
;;     6114.954437060316 1,
;;     6127.051822384062 1,
;;     6111.873837691539 1,
;;     6698.007896613374 1,
;;     6122.605644139096 1,
;;     6119.914132071367 1,
;;     6134.204694728398 1,
;;     6158.498740021148 1,
;;     6127.590085751611 1,
;;     6122.764197188769 1,
;;     6143.974670366456 1,
;;     6135.659227365713 1,
;;     6121.0155736500055 1,
;;     6648.1538394675445 1,
;;     6135.356402544607 1,
;;     6115.265812165857 1,
;;     6143.272997115649 1,
;;     6142.195498948722 1,
;;     6123.488759735313 1,
;;     6117.175849113254 1,
;;     6128.670214294472 1,
;;     6122.705800691573 1,
;;     6640.362507621006 1,
;;     6114.22936715954 2,
;;     6116.0339587805665 1,
;;     6622.440785807742 1,
;;     6644.097035906398 1,
;;     6113.2835977525865 1,
;;     6629.8793804672 1,
;;     6119.986119015085 1,
;;     6110.811827733896 1,
;;     6123.811223445067 1}


;; => {6123.890473823705 1,
;;     6669.880120977151 1,
;;     6125.2824247996905 1,
;;     6640.707385218701 1,
;;     6667.623575206193 1,
;;     3113.1273632925795 92,
;;     6132.120814717038 1,
;;     6121.701204337272 1,
;;     6635.601704747361 1}




;; # 5. in one threading macro, no variables declared
;; same as 4., but written more compact

(def distortion-2
  (->> [[:mm/std-scale :all {}]
        {:metamorph/id :k-means}
        [:scicloj.ml.smile.clustering/cluster
         :k-means
         [3 300]
         :cluster]]
       ml/->pipeline
       (ml/fit-pipe data)
       :k-means
       :info))

(println :scicloj-ml-decl-2 distortion-2)
