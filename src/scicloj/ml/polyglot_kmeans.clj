(ns scicloj.ml.polyglot-kmeans
  (:require
   [scicloj.sklearn-clj.metamorph]
   [nextjournal.clerk :as clerk]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :as py :refer [py.- py.]]))

(comment
  (clerk/serve! {})
  (clerk/build-static-app! {:paths ["src/scicloj/ml/polyglot_kmeans.clj"]
                            :bundle? false})
  (clerk/clear-cache!))

;; The following python code can be expressed  as Clojure in
;; 4 different ways

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

;; # 2. use sklearn-clj
;; This librraies allow to use all estimaots/model from sklearn
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

(-> fitted-ctx-2 :k-means  :info)

;; # 4. use declarative Clojure only pipeline
;; same as 3), only using metamorph declarative pipelines



(def decl-pipe
  [[:mm/std-scale :all {}]
   {:metamorph/id :k-means}
   [:scicloj.ml.smile.clustering/cluster
    :k-means
    [3 300]
    :cluster]])

(->> decl-pipe
     ml/->pipeline
     (ml/fit-pipe data)
     :k-means
     :info)
     






;; # 5. in one threading macro, no variables declared
;; same as 4., but written more compact


(->> [[:mm/std-scale :all {}]
      {:metamorph/id :k-means}
      [:scicloj.ml.smile.clustering/cluster
       :k-means
       [3 300]
       :cluster]]
     ml/->pipeline
     (ml/fit-pipe data)
     :k-means
     :info)
     
