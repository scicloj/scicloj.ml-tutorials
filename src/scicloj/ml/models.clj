^{:nextjournal.clerk/visibility #{:hide-ns :hide}}
(ns scicloj.ml.models
  (:require
   [tablecloth.api :as tc]
   [scicloj.ml.core]
   [scicloj.sklearn-clj.ml]
   [clojure.string :as str]
   [scicloj.ml.ug-utils :refer :all]
   [scicloj.ml.ug-utils-clerk :as utils-clerk]
   [clojure.java.io :as io]
   [nextjournal.clerk :as clerk]))


^{:nextjournal.clerk/viewer :hide-result}
(comment
  (clerk/show! "src/scicloj/ml/models.clj")
  (clerk/halt!)
  (clerk/build-static-app! {:paths ["src/scicloj/ml/models.clj"]
                            :bundle? false})
  (clerk/clear-cache!)
  (clerk/serve! {:browse? true})
  (clerk/serve! {:browse? true :watch-paths ["src/scicloj/ml/"]}))



^{:nextjournal.clerk/viewer :hide-result}
(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset :refer [dataset add-column]]
         '[scicloj.ml.dataset :as ds]
         '[tech.v3.dataset.math :as std-math]
         '[tech.v3.datatype.functional :as dtf]
         '[scicloj.metamorph.ml.toydata :as datasets])


(clerk/set-viewers! [{:pred tc/dataset?
                      :transform-fn #(clerk/table {:head (ds/column-names %)
                                                   :rows (ds/rows % :as-seq)})}])


^{:nextjournal.clerk/viewer :hide-result}
(def build-in-models
  (->>
   (ml/model-definition-names)
   (filter #(contains? #{"fastmath.cluster"
                         "smile.classification"
                         "smile.regression"
                         "smile.manifold"
                         "smile.projections"
                         "xgboost"}
                       (namespace %)))
   sort))

(defn make-iris-pipeline [model-options]
  (ml/pipeline
   (mm/set-inference-target :species)
   (mm/categorical->number [:species])
   (mm/model model-options)))


;; # Models

;; scicloj.ml uses the plugin `scicloj.ml.smile` and
;; `scicloj.ml.xgboost` by default,
;; which gives access to " (count build-in-models) " models from the java libraries
;; [Smile](https://haifengl.github.io/),
;; [Xgboost](https://xgboost.readthedocs.io/en/latest/jvm/index.html) and [fastmath](https://github.com/generateme/fastmath)

;; More models are avilable via other plugins.

;; Below is a list of all such models, and which parameter they take.

;; All models are available in the same way:



;; The documentation below points as well to the javadoc and user-guide chapter (for Smile models)

;; The full list of build in models is:

(clerk/html
 [:ul

  (map
   #(vector :li [:a {:href (str "#" (str %))} (str %)])
   build-in-models)])


;; ## Smile classification models

(clerk/html
 (utils-clerk/render-key-info ":smile.classification/ada-boost"))


;; In this example we will use the capability of the Ada boost classifier
;; to give us the importance of variables.

;; As data we take here the Wiscon Breast Cancer dataset, which has 30 variables.

(def df
  (->
   (datasets/breast-cancer-ds)))





;; To get an overview of the dataset, we print its summary:



(-> df ds/info)


;; Then we create a metamorph  pipeline with the ada boost model:

(def ada-pipe-fn
  (ml/pipeline
   (mm/set-inference-target :class)
   (mm/categorical->number [:class])
   (mm/model
    {:model-type :smile.classification/ada-boost})))


;; We run the pipeline in :fit. As we just explore the data,
;; not train.test split is needed.

(def trained-ctx
  (ml/fit-pipe df
   ada-pipe-fn))

;; "Next we take the model out of the pipeline:"
(def model
  (-> trained-ctx vals (nth 2) ml/thaw-model))

;; The variable importance can be obtained from the trained model,
(def var-importances
  (mapv
   #(hash-map :variable %1
              :importance %2)
   (map
    #(first (.variables %))
    (.. model formula predictors))
   (.importance model)))


;; and we plot the variables:

(clerk/vl
 {
  :data {:values
          var-importances}
  :width  800
  :height 500
  :mark {:type "bar"}
  :encoding {:x {:field :variable :type "nominal" :sort "-y"}
             :y {:field :importance :type "quantitative"}}})



(clerk/html
 (utils-clerk/render-key-info ":smile.classification/decision-tree"))

;; A decision tree learns a set of rules from the data in the form
;; of a tree, which we will plot in this example.
;; We use the iris dataset:


(def iris  ^:nextjournal.clerk/no-cache  (datasets/iris-ds))

iris

;; We make a pipe only containing the model, as the dataset is ready to
;; be used by `scicloj.ml`
(def trained-pipe
  (ml/fit-pipe
   iris
   (ml/pipeline
    {:metamorph/id :model}
    (mm/model
     {:model-type :smile.classification/decision-tree}))))

;; We extract the Java object of the trained model.

(def tree-model
  (-> trained-pipe :model ml/thaw-model))


;; The model has a .dot function, which returns a GraphViz textual
;; representation of the decision tree, which we render to svg using the
;; [kroki](https://kroki.io/) service.

(clerk/html
 (String. (:body (kroki (.dot tree-model) :graphviz :svg)) "UTF-8"))


(clerk/html (utils-clerk/render-key-info ":smile.classification/discrete-naive-bayes"))

(clerk/html (utils-clerk/render-key-info ":smile.classification/gradient-tree-boost"))

(clerk/html (utils-clerk/render-key-info ":smile.classification/knn"))
;; In this example we use a knn model to classify some dummy data.
;; The training data is this:

(def df-knn
  (ds/dataset {:x1 [7 7 3 1]
               :x2 [7 4 4 4]
               :y [ :bad :bad :good :good]}))



;; Then we construct a pipeline with the knn model,
;; using 3 neighbors for decision.

(def knn-pipe-fn
  (ml/pipeline
   (mm/set-inference-target :y)
   (mm/categorical->number [:y])
   (mm/model
    {:model-type :smile.classification/knn
     :k 3})))

;; We run the pipeline in mode fit:

(def trained-ctx-knn
  (knn-pipe-fn {:metamorph/data df-knn
                :metamorph/mode :fit}))


;; Then we run the pipeline in mode :transform with some test data
;; and take the prediction and convert it from numeric into categorical:

(->
 trained-ctx-knn
 (merge
  {:metamorph/data (ds/dataset
                    {:x1 [3 5]
                     :x2 [7 5]
                     :y [nil nil]})
   :metamorph/mode :transform})
 knn-pipe-fn
 :metamorph/data
 (ds/column-values->categorical :y)
 seq)


(clerk/html (utils-clerk/render-key-info ":smile.classification/logistic-regression"))

(clerk/html (utils-clerk/render-key-info ":smile.classification/maxent-binomial"))

(clerk/html (utils-clerk/render-key-info ":smile.classification/maxent-multinomial"))

(clerk/html (utils-clerk/render-key-info ":smile.classification/random-forest"))
;; The following code plots the decision surfaces of the random forest
 ;; model on pairs of features.

;; We use the Iris dataset for this.

(def iris-test
  (ds/dataset
   "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword}))




;; Standarise the data:
(def iris-std
  (ml/pipe-it
   iris-test
   (mm/std-scale [:sepal_length :sepal_width :petal_length :petal_width] {})))






;; The next function creates a vega specification for the random forest
;; decision surface for a given pair of column names.




(def rf-pipe
  (make-iris-pipeline
                      {:model-type :smile.classification/random-forest}))

(clerk/vl (surface-plot iris [:sepal_length :sepal_width] rf-pipe :smile.classification/random-forest))

(clerk/vl
 (surface-plot iris-std [:sepal_length :petal_length] rf-pipe :smile.classification/random-forest))

(clerk/vl
 (surface-plot iris-std [:sepal_length :petal_width] rf-pipe :smile.classification/random-forest))
(clerk/vl
 (surface-plot iris-std [:sepal_width :petal_length] rf-pipe :smile.classification/random-forest))
(clerk/vl
 (surface-plot iris-std [:sepal_width :petal_width] rf-pipe :smile.classification/random-forest))
(clerk/vl
 (surface-plot iris-std [:petal_length :petal_width] rf-pipe :smile.classification/random-forest))


(clerk/html (utils-clerk/render-key-info ":smile.classification/sparse-logistic-regression"))
(clerk/html (utils-clerk/render-key-info ":smile.classification/sparse-svm"))
(clerk/html (utils-clerk/render-key-info ":smile.classification/svm"))

;; ## Smile regression

(clerk/html (utils-clerk/render-key-info ":smile.regression/elastic-net"))


(clerk/html (utils-clerk/render-key-info ":smile.regression/gradient-tree-boost"))


(clerk/html (utils-clerk/render-key-info ":smile.regression/lasso"))

;; We use the diabetes dataset and will show how Lasso regression
;; regulates the different variables dependent of lambda.

;; First we make a function to create pipelines with different lambdas
(defn make-pipe-fn [lambda]
  (ml/pipeline
   (mm/update-column :disease-progression (fn [col] (map #(double %) col)))
   (mm/convert-types :disease-progression :float32)
   (mm/set-inference-target :disease-progression)
   {:metamorph/id :model} (mm/model {:model-type :smile.regression/lasso
                                     :lambda (double lambda)})))

;; No we go over a sequence of lambdas and fit a pipeline for all off them
;; and store the coefficients for each predictor variable:
(def diabetes (datasets/diabetes-ds))

(def coefs-vs-lambda
  (flatten
   (map
    (fn [lambda]
      (let [fitted
            (ml/fit-pipe
             diabetes
             (make-pipe-fn lambda))

            model-instance
            (-> fitted
                :model
                (ml/thaw-model))

            predictors
            (map
             #(first (.variables %))
             (seq
              (.. model-instance formula predictors)))]

        (map
         #(hash-map :log-lambda (dtf/log10 lambda)
                    :coefficient %1
                    :predictor %2)
         (-> model-instance .coefficients seq)
         predictors)))
    (range 1 100000 100))))

;; Then we plot the coefficients over the log of lambda.
(clerk/vl
 {
  :data {:values coefs-vs-lambda}

  :width 500
  :height 500
  :mark {:type "line"}
  :encoding {:x {:field :log-lambda :type "quantitative"}
             :y {:field :coefficient :type "quantitative"}
             :color {:field :predictor}}})

;; This shows that an increasing lambda regulates more and more variables
 ;; to zero. This plot can be used as well to find important variables,
;; namely the ones which stay > 0 even with large lambda.

(clerk/html
 (utils-clerk/render-key-info ":smile.regression/ordinary-least-square"))

;; In this example we will explore the relationship between the
;; body mass index (bmi) and a diabetes indicator.

;; First we load the data and split into train and test sets.
;;
^{:nextjournal.clerk/viewer :hide-result}
(def diabetes (datasets/diabetes-ds))

^{:nextjournal.clerk/viewer :hide-result}
(def diabetes-train
  (ds/head diabetes 422))

^{:nextjournal.clerk/viewer :hide-result}
(def diabetes-test
  (ds/tail diabetes 20))



;; Next we create the pipeline, converting the target variable to
;; a float value, as needed by the model.

(def ols-pipe-fn
  (ml/pipeline
   (mm/select-columns [:bmi :disease-progression])
   (mm/convert-types :disease-progression :float32)
   (mm/set-inference-target :disease-progression)
   {:metamorph/id :model} (mm/model {:model-type :smile.regression/ordinary-least-square})))

;; We can then fit the model, by running the pipeline in mode :fit

(def fitted
  (ml/fit diabetes-train ols-pipe-fn))


;; Next we run the pipe-fn in :transform and extract the prediction
;; for the disease progression:
(def diabetes-test-prediction
  (-> diabetes-test
      (ml/transform-pipe ols-pipe-fn fitted)
      :metamorph/data
      :disease-progression))

;; The truth is available in the test dataset.
(def diabetes-test-trueth
  (-> diabetes-test
      :disease-progression))




;; The smile Java object of the LinearModel is in the pipeline as well:

(def model-instance
  (-> fitted :model  (ml/thaw-model)))

;; This object contains all information regarding the model fit
;; such as coefficients and formula:
(-> model-instance .coefficients seq)
(-> model-instance .formula str)

;; Smile generates as well a String with the result of the linear
;; regression as part of the toString() method of class LinearModel:

(clerk/code
 (str model-instance))



;; This tells us that there is a statistically significant
;; (positive) correlation between the bmi and the diabetes
;; disease progression in this data.


;; At the end we can plot the truth and the prediction on the test data,
;; and observe the linear nature of the model.

(clerk/vl
 {:layer [
          {:data {:values (map #(hash-map :disease-progression %1 :bmi %2 :type :truth)
                               diabetes-test-trueth
                               (:bmi  diabetes-test))}

           :width 500
           :height 500
           :mark {:type "circle"}
           :encoding {:x {:field :bmi :type "quantitative"}
                      :y {:field :disease-progression :type "quantitative"}
                      :color {:field :type}}}

          {:data {:values (map #(hash-map :disease-progression %1 :bmi %2 :type :prediction)
                               diabetes-test-prediction
                               (:bmi diabetes-test))}

           :width 500
           :height 500
           :mark {:type "line"}
           :encoding {:x {:field :bmi :type "quantitative"}
                      :y {:field :disease-progression :type "quantitative"}
                      :color {:field :type}}}]})



(clerk/html (utils-clerk/render-key-info ":smile.regression/random-forest"))

(clerk/html (utils-clerk/render-key-info ":smile.regression/ridge"))


;; ## Xgboost

(clerk/html (utils-clerk/render-key-info ":xgboost"))

;; ## fastmath.cluster

(clerk/html (utils-clerk/render-key-info :fastmath.cluster))

;; ## smile.projections

(clerk/html (utils-clerk/render-key-info :smile.projections))

;; ## smile.manifold

(clerk/html (utils-clerk/render-key-info :smile.manifold))


;; # Compare decision surfaces of different models

;; In the following we see the decision surfaces of some models on the
;; same data from the Iris dataset using 2 columns :sepal_width and sepal_length:

(map #(clerk/vl (surface-plot iris-std [:sepal_length :sepal_width] (make-iris-pipeline %)) (:model-type %))
     [
      {:model-type :smile.classification/ada-boost}
      {:model-type :smile.classification/decision-tree}
      {:model-type :smile.classification/gradient-tree-boost}
      {:model-type :smile.classification/knn}
      {:model-type :smile.classification/logistic-regression}
      {:model-type :smile.classification/random-forest}
      {:model-type :smile.classification/linear-discriminant-analysis}
      {:model-type :smile.classification/regularized-discriminant-analysis}
      {:model-type :smile.classification/quadratic-discriminant-analysis}
      {:model-type :xgboost/classification}])



;; This shows nicely that different model types have different capabilities
;; seperate and tehre fore classify data.


;; ## Ensembles

;; An ensemble is combining several pipelines and their prediction
;; and calculate a common prediction.
;; `sicloj.ml` alows to create an ensemble whehre each model gives avote,
;; and the majority becomes the final prediction.
;;


;;  First we make three pipelines, which only differ in the model type.
;;  The pipleines could b completely different, but need to accept the same input data and
;;  produce the same predictions (target column name and type)
;;
(defn make-iris-pipeline [model-type]
  (ml/pipeline
   (mm/select-columns [:species :sepal_length :sepal_width])
   (mm/set-inference-target :species)
   (mm/categorical->number [:species])

   {:metamorph/id :model}
   (mm/model
    {:model-type model-type})))

(def tree-pipeline
  (make-iris-pipeline :smile.classification/decision-tree))
 

(def knn-pipeline
  (make-iris-pipeline :smile.classification/knn))
  

(def logistic-regression-pipeline
  (make-iris-pipeline :smile.classification/logistic-regression))
  

;;  Know we can contruct an enseme, using function `ensemble-pipe`

(def ensemble (ml/ensemble-pipe [tree-pipeline
                                 knn-pipeline
                                 logistic-regression-pipeline]))

;;  thos ensemble is as any other metamorh pipeline,
;;  so we can train and predict as usual:

(def fitted-ctx-ensemble
  (ml/fit-pipe iris-std ensemble))

(def transformed-ctx-ensemble
  (ml/transform-pipe iris-std ensemble fitted-ctx-ensemble))


;;  Frequency of predictions
(->
 transformed-ctx-ensemble
 :metamorph/data
 (ds/reverse-map-categorical-xforms)
 :species
 frequencies)

;;  The surface plot of the ensemble

(clerk/vl (surface-plot iris-std
                            [:sepal_length :sepal_width]
                            ensemble "voting ensemble"))






