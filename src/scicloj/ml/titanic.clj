(ns scicloj.ml.titanic
  (:require
   [notespace.api :as note]
   [notespace.kinds :as kind]))

(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/userguide-titanic.html")

  (note/init))
  


(require '[scicloj.ml.dataset :as ds]
         '[tech.v3.dataset.math :as ds-math]
         '[tech.v3.datatype.functional :as dfn]
         '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[camel-snake-kebab.core :as csk]
         '[scicloj.metamorph.ml.loss :as loss]
         '[clojure.string :as str]
         '[fastmath.stats :as stats]
         '[fastmath.random :as rnd]
         '[scicloj.ml.xgboost] :reload)
          


["## Introduction "]

[" In this example, we will train a model which is able to predict the survival of passengers from the Titanic dataset."
 "In a real analysis, this would contain as well explorative analysis of the data, which I will skip here,
as the purpose is to showcase machine learning with scicloj.ml, which is about model evaluation and selection."]
 


["### Read data"]

(def data (ds/dataset "data/titanic/train.csv" {:key-fn csk/->kebab-case-keyword}))



["Column info:"]
(ds/info data)


["We can explore the association between the categorical columns of the dataset
with the :survived using cramers-v-corrected:"]
(def categorical-feature-columns [:pclass :sex :age :parch
                                    :embarked])
(map
 #(hash-map
   %
   (stats/cramers-v-corrected
    (get  data %)
    (:survived data)))
 categorical-feature-columns)
 
["In this dataset, :sex seems to be the best predictor for survival."]

["Association between the select variables:"]
(for [c1 categorical-feature-columns c2 categorical-feature-columns]
  {[c1 c2]
   (stats/cramers-v-corrected (get data c1) (get data  c2))})
  

["This shows how much the columns are correlated. "]

["## clean some of the features"]

["The follwoing functios will be used in the pipeline. They clean the
features to make them better predictors."]

(defn categorize-cabin [data]
  (-> data
      (ds/add-or-replace-column
       :cabin
       (map
        #(if (empty? %)
           :unknown
           (keyword (subs
                     %
                     0 1)))
        (:cabin data)))))
        

(defn categorize-age [data]
  (->
   data
   (ds/add-or-replace-column
    :age-group
    (map
     #(cond
        (< % 10) :child
        (< % 18) :teen
        (< % 60) :adult
        (> % 60) :elderly
        true :other)
     (:age data)))))

["We want to create a new column :title which might help in the score.
This is an example of custom function, which creates a new column from existing columns,
which is a typical case of feature engineering."]

(defn name->title [dataset]
  (-> dataset
      (ds/add-or-replace-column
       :title
       (map
        #(-> % (str/split  #"\.")
             first
             (str/split  #"\,")
             last
             str/trim)
        (data :name)))
      (ds/drop-columns :name)))

(def title-map
  {"Major" :a
   "Col" :a
   "Rev" :a
   "Ms" :b
   "Miss" :b
   "Jonkheer" :a
   "Don" :a
   "Mlle" :b
   "Mr" :a
   "Master" :a
   "Capt" :a
   "Mrs" :b
   "Lady" :b
   "Sir" :a
   "Dr" :a
   "the Countess" :b
   "Mme" :b})

(defn categorize-title [data]
 (->
    data
    (ds/add-or-replace-column
     :title
     (map title-map (:title data)))))

["The final pipeline contains the functions we did before."]


;; => _unnamed [2 1]:
;;    | :a |
;;    |----|
;;    |    |
;;    |    |

(def pipeline-fn
  (ml/pipeline
   (mm/replace-missing :embarked :value "S")
   (mm/replace-missing :age :value tech.v3.datatype.functional/mean)
   (mm/update-column :parch str)
   (ml/lift categorize-age)
   (ml/lift name->title)
   (ml/lift categorize-title)
   (ml/lift categorize-cabin)
   (mm/select-columns [:age-group
                       :cabin
                       :embarked
                       :fare
                       :parch
                       :pclass
                       :sex
                       :survived
                       :title])

   (fn [ctx]
     (assoc ctx :categorical-ds
            (:metamorph/data ctx)))


   (mm/categorical->number [:survived :pclass :sex :embarked
                            :title :age-group :cabin :parch] {} :int64)

   (mm/set-inference-target :survived)))


["Transformed data"]
(->
 (pipeline-fn {:metamorph/data data :metamorph/mode :fit})
 :metamorph/data)


["The following splits the dataset in three pieces,
 train, val and test to predict on later.
"]





(def ds-split (first (ds/split->seq data :holdout {:ratio [0.8 0.2]
                                                   :split-names [:train-val :test]})))
                                    

["Create a sequence of train/test  (k-fold with k=10) splits used to evaluate the pipeline."]
(def train-val-splits
    (ds/split->seq
     (:train-val ds-split)
     :kfold
     {:k 10}))




["The full pipeline definition including the random forrest model."]

(def full-pipeline-fn
  (ml/pipeline
   pipeline-fn
   ;; we overwrite the id, so the model function will store
   ;; it's output (the model) in the pipeline ctx under key :model
   {:metamorph/id :model}
   (mm/model {:model-type :smile.classification/random-forest})))





["Evaluate the (single) pipeline function using the train/test split"]
(def evaluations
  (ml/evaluate-pipelines
   [full-pipeline-fn]
   train-val-splits
   ml/classification-accuracy
   :accuracy))


["The default k-fold splits makes 10 folds,
so we train 10 models, each having its own loss."]

["The `evaluate-pipelines` fn averages the models per pipe-fn,
and returns the best.
So we get a single model back, as we only have one pipe fn"]

["Often we consider the model with the lowest loss to be the best."]

["Return a single model only (as a list of 1) , namely the best over all
 pipeline functions
and all cross validations is the default behavoiur, but can be changed
with the `tune options`."]

["They controll as well which information is returned."]

["`tech.ml` stores the models in the context in a serialzed form,
and the function `thaw-model` can be used to get the original model back.
This is a Java class in the case of
 model :smile.classification/random.forest, but this depends on the
which `model` function is in the pipeline"]

["We can get for example,  the models like this:"]

(def models
  (->> evaluations
       flatten
       (map
        #(hash-map :model (ml/thaw-model (get-in % [:fit-ctx :model]))
                   :metric ((comp :metric :test-transform) %)
                   :fit-ctx (:fit-ctx %)))
                   
       (sort-by :mean)
       reverse))


["The accuracy of the best trained model is:"]
(-> models first :metric)

["The one with the highest accuracy is then:"]
(-> models first :model)


["We can get the predictions on new-data, which for classification contain as well
the posterior probabilities per class."]

["We do this by running the pipeline again, this time with new data and merging
:mode transform"]

(def predictions
  (->
   (full-pipeline-fn
    (assoc
     (:fit-ctx (first models))
     :metamorph/data (:test ds-split)
     :metamorph/mode :transform))
   :metamorph/data))

^kind/dataset
predictions


["Out of the predictions and the truth, we can construct the
 confusion matrix."]

(def trueth
  (->
   (full-pipeline-fn {:metamorph/data (:test ds-split) :metamorph/mode :fit})
   :metamorph/data
   tech.v3.dataset.modelling/labels))

^kind/dataset
(->
 (ml/confusion-map (:survived predictions)
                   (:survived trueth)
                   :none)
 (ml/confusion-map->ds))

["### Hyper parameter tuning"]

["This defines a pipeline with options. The options gets passed to the model function,
so become hyper-parameters of the model.

The `use-age?` options is used to make a conditional pipeline. As the use-age? variable becomes part of the grid to search in,
we tune it as well.
This is an example how pipeline-options can be grid searched in the same way then hyper-parameters of the model.

"]
(defn make-pipeline-fn [options]

  (ml/pipeline
   pipeline-fn
   {:metamorph/id :model}
   (mm/model
    (merge options
           {:model-type :smile.classification/random-forest}))))

["Use sobol optimization, to find som grid points,
which cover in a smart way the hyper-parameter space."]

(def search-grid
  (->>
   (ml/sobol-gridsearch {:trees (ml/linear 100 500 10)
                         :mtry (ml/categorical [0 2 4])
                         :split-rule (ml/categorical [:gini :entropy])
                         :max-depth (ml/linear 1 50 10)
                         :node-size (ml/linear 1 10 10)})

   (take 500)))
  

["Generate the pipeline-fns we want to evaluate."]
(def pipeline-fns (map make-pipeline-fn search-grid))

(defn xgboost-pipe [opts]
  (ml/pipeline
     pipeline-fn
     {:metamorph/id :model}
     (mm/model
      (merge opts
             {:model-type :xgboost/classification}))))

(def xgboost-pipes
  (->>
   (ml/sobol-gridsearch
    (ml/hyperparameters :xgboost/classification))
   (take 500)
   (map xgboost-pipe)))


;; (ml/fit-pipe (:train (first train-val-splits)) xgboost-pipe)

["Evaluate all  pipelines and keep results"]
(def evaluations

  (ml/evaluate-pipelines
   (take 10
         (concat xgboost-pipes xgboost-pipes))
   train-val-splits
   ml/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false
    :return-best-crossvalidation-only false
    ;; :evaluation-handler-fn (fn [m]
    ;;                          (println (:metric m)))


    :map-fn :map
    :result-dissoc-seq []}))
    





["Get the key information from the evaluations and sort by the metric function used,
 accuracy here."]

(def models
  (->> evaluations
       flatten
       (map
        #(assoc
          (select-keys % [:test-transform :fit-ctx :pipe-fn])

          :model (ml/thaw-model (get-in % [:fit-ctx :model]))))
       (sort-by (comp :metric :test-transform))
       reverse))




["As we did several pipelines and several x-fold cross validation, we have quite some models trained in total "]
(count models)

["As we sorted by mean accuracy, the first evaluation result is the best model,"]
(def best-model (first models))

["which is: "]
(:model best-model)

["with a mean accuracy of "  (-> best-model :test-transform :mean)]
["and a accuracy of "  (-> best-model :test-transform :metric)]


(println "mean acc: " (-> best-model :test-transform :mean))
(println "acc: " (-> best-model :test-transform :metric))


["using options: "]
(-> best-model :fit-ctx :model :options)
(clojure.pprint/pprint (-> best-model :fit-ctx :model :options))

(def test-data (ds/dataset "data/titanic/test.csv"
                           {:key-fn csk/->kebab-case-keyword}))



(def predition-on-test
  (full-pipeline-fn
   (assoc (:fit-ctx best-model)
          :metamorph/data (ds/add-column test-data :survived nil)
          :metamorph/mode :transform)))


(def prediction-ds
  (->
   (predition-on-test :metamorph/data)
   (ds/add-column :passenger-id (:passenger-id test-data))
   (ds/convert-types [:survived] :int)
   (ds/select-columns [:passenger-id :survived 0 1])))

^kind/dataset
prediction-ds





["# Create Subimssion file to Kaggle"]

(def submission-ds
  (-> prediction-ds
      (ds/select-columns [:passenger-id :survived])
      (ds/rename-columns {:passenger-id "PassengerId"
                          :survived "Survived"})))

(ds/write-csv! submission-ds "submission.csv")


["### Learning curve"]



(def training-curve-splits
  (map
   #(hash-map :train (ds/head (:train-val ds-split) %)
              :test (:test ds-split))
   (range 5 (ds/row-count (:train-val ds-split)) 10)))



(def training-curve-evaluations
  (ml/evaluate-pipelines [(:pipe-fn (first models))]
                         training-curve-splits
                         ml/classification-accuracy
                         :accuracy
                         {:map-fn :map
                          :return-best-pipeline-only false
                          :return-best-crossvalidation-only false
                          :result-dissoc-in-seq []}))
(def train-counts
  (->> training-curve-evaluations flatten (map #(-> % :fit-ctx :metamorph/data ds/row-count))))



(def test-metrices
  (->> training-curve-evaluations flatten (map #(-> % :test-transform :metric))))

(def train-metrices
  (->> training-curve-evaluations flatten (map #(-> % :train-transform :metric))))

(def traing-curve-plot-data
  (reverse
   (sort-by :metric
            (flatten
             (map
              #(vector (zipmap [:count :metric :type] [%1 %2 :test])
                       (zipmap [:count :metric :type] [%1 %3 :train]))
              train-counts
              test-metrices
              train-metrices)))))


^kind/vega
{
 :data {:values traing-curve-plot-data}

 :width 500
 :height 500
 :mark {:type "line"}
 :encoding {:x {:field :count :type "quantitative"}
            :y {:field :metric :type "quantitative"}
            :color {:field :type}}}





(comment
  (->>
   (map
    #(hash-map :test-metric %1
               :train-metric %2
               :better? (if (> %1 %2) :test :train))
    (->> training-curve-evaluations flatten (map :metric))
    (->> training-curve-evaluations flatten (map #(get-in % [:train-prediction :metric]))))
   (map :better?)
   frequencies)

  (println
   (-> (ds/dataset {:x ["A" "B" "C" "D" "E" "F"] :y (range)})
       (ds/categorical->one-hot [:x] {} :int)
       (ds/set-inference-target :y)
       (scicloj.metamorph.ml/train {:model-type :smile.regression/ordinary-least-square})
       ml/thaw-model)))

  
