(ns _05-15-regression
  (:require [appliedsciencestudio.rdata :as rdata]
            [tech.v3.datatype.functional :as func]
            [scicloj.ml.dataset :as ds]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.core :as ml]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]
            [aerial.hanami.core :as hmi]
            [nextjournal.clerk.viewer :as v]
            [camel-snake-kebab.core :as csk]
            [regression-plot :as reg-plot]))

          
(comment
  (require '[notespace.api :as note])
  (require '[notespace.kinds :as kinds])

  (notespace.api/update-config
   #(assoc % :source-base-path "notebook"))
  (note/update-config)
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html "docs/userguide-advanced.html")
  (note/init))

(comment


  (require

   '[nextjournal.clerk.webserver :as webserver]
   '[nextjournal.clerk :as clerk]
   '[nextjournal.beholder :as beholder])



  (webserver/start! {:port 7777})

  (def filewatcher
    (beholder/watch #(clerk/file-event %) "notebook"))

  (clerk/show! "notebook/_05_15_regression.clj")
  (clerk/clear-cache!)
  :ok)


(defn valid-kebap-case-keyword [s]
  (-> s
      (clojure.string/replace \. \-)
      csk/->kebab-case-keyword))

(def data
  (->>
   (rdata/read-rdata "data/parenthood.Rdata" {:key-fn valid-kebap-case-keyword})
   :parenthood
   ds/dataset))

(def lm
  (-> data
      (ds/select-columns [:dan-grump :dan-sleep])
      (ds/set-inference-target :dan-grump)
      (scicloj.metamorph.ml/train {:model-type :smile.regression/ordinary-least-square})))
(seq
 (.coefficients
  (ml/thaw-model lm)))



(def line-data (ds/dataset {:dan-sleep (range 4 9 0.1)}))

(def predictions-ds
  (-> line-data
      (ml/predict  lm)
      (ds/append line-data)))


(def predictions
  (-> predictions-ds
      (ds/rows :as-maps)
      seq))
      


;;
;;  https://learningstatisticswithr.com/book/index.html
;;  https://github.com/djnavarro/rbook
;;
;; # Linear regression{#regression}

;; The goal in this chapter is to introduce **_linear regression_**, the standard tool that statisticians rely on when analysing
;; the relationship between interval scale predictors and interval scale outcomes. Stripped to its bare essentials, linear regression
;; models are basically a slightly fancier version of the Pearson correlation (Section \@ref(correl)) though as we'll see, regression models
;; are much more powerful tools.

;; ## What is a linear regression model?






^kinds/vega
(hc/xform reg-plot/regression-points-chart

         :VALDATA (seq (ds/rows data :as-maps))
         :X :dan-sleep :Y :dan-grump
         :MCOLOR :black
         :XTITLE "My sleep (hours)"
         :YTITLE "My grumpiness (0-100)"
         :XSCALE {"zero" false}
         :YSCALE {"zero" false})



^kinds/vega
(hc/xform reg-plot/regression-chart
           :TITLE "The best fitting regression line "
           :REGRESSION-X :dan-sleep
           :REGRESSION-Y :dan-grump
           :REGRESSION-POINT-DATA (seq (ds/rows data :as-maps))
           :REGRESSION-LINE-DATA predictions)




;; Figure 15.2: Panel a shows the sleep-grumpiness scatterplot from above with the best fitting regression line drawn over the top. Not surprisingly, the line goes through the middle of the data.

(def bad-line-data
  (map #(hash-map :dan-sleep %1
                  :dan-grump %2)
       (range 4 9 0.1)
       (func/+  80
                (func/* -3
                        (range 4 9 0.1)))))
^kinds/vega
(hc/xform reg-plot/regression-chart
          :TITLE "Not the best fitting regression line"
          :REGRESSION-X :dan-sleep
           :REGRESSION-Y :dan-grump

          :REGRESSION-POINT-DATA (seq (ds/rows data :as-maps))
          :REGRESSION-LINE-DATA bad-line-data)
;; Figure 15.3: In contrast, this plot shows the same data, but with a very poor choice of regression line drawn over the top.


;; Since the basic ideas in regression are closely tied to correlation, we’ll return to the parenthood.Rdata file that we were using to illustrate how correlations work. Recall that, in this data set, we were trying to find out why Dan is so very grumpy all the time, and our working hypothesis was that I’m not getting enough sleep. We drew some scatterplots to help us examine the relationship between the amount of sleep I get, and my grumpiness the following day. The actual scatterplot that we draw is the one shown in Figure 15.1, and as we saw previously this corresponds to a correlation of
;; r=−.90
;; , but what we find ourselves secretly imagining is something that looks closer to Figure 15.2. That is, we mentally draw a straight line through the middle of the data. In statistics, this line that we’re drawing is called a regression line. Notice that – since we’re not idiots – the regression line goes through the middle of the data. We don’t find ourselves imagining anything like the rather silly plot shown in Figure 15.3.

;; This is not highly surprising: the line that I’ve drawn in Figure 15.3 doesn’t “fit” the data very well, so it doesn’t make a lot of sense to propose it as a way of summarising the data, right? This is a very simple observation to make, but it turns out to be very powerful when we start trying to wrap just a little bit of maths around it. To do so, let’s start with a refresher of some high school maths. The formula for a straight line is usually written like this:
;; y=mx+c

;; Or, at least, that’s what it was when I went to high school all those years ago. The two variables are
;; x
;; and
;; y
;; , and we have two coefficients,
;; m
;; and
;; c
;; . The coefficient
;; m
;; represents the slope of the line, and the coefficient
;; c
;; represents the
;; y
;; -intercept of the line. Digging further back into our decaying memories of high school (sorry, for some of us high school was a long time ago), we remember that the intercept is interpreted as “the value of
;; y
;; that you get when
;; x = 0
;; . Similarly, a slope of
;; m
;; means that if you increase the
;; x
;; -value by 1 unit, then the
;; y
;; -value goes up by
;; m
;; units; a negative slope means that the
;; y
;; -value would go down rather than up. Ah yes, it’s all coming back to me now.

;; Now that we’ve remembered that, it should come as no surprise to discover that we use the exact same formula to describe a regression line. If
;; Y
;; is the outcome variable (the DV) and
;; X
;; is the predictor variable (the IV), then the formula that describes our regression is written like this:
;; ^ Y i = b 1 X i + b 0

;; Hm. Looks like the same formula, but there’s some extra frilly bits in this version. Let’s make sure we understand them. Firstly, notice that I’ve written
;; X i
;; and
;; Y i
;; rather than just plain old
;; X
;; and
;; Y
;; . This is because we want to remember that we’re dealing with actual data. In this equation,
;; X i
;; is the value of predictor variable for the
;; i
;; th observation (i.e., the number of hours of sleep that I got on day
               ;; i
               ;; of my little study
     ;; , and

;; Y i
;; is the corresponding value of the outcome variable (i.e., my grumpiness on that day). And although I haven’t said so explicitly in the equation, what we’re assuming is that this formula works for all observations in the data set (i.e., for all
                           ;; i

 ;; . Secondly, notice that I wrote

;; ^ Y i
;; and not
;; Y i
;; . This is because we want to make the distinction between the actual data
;; Y i
;; , and the estimate
;; ^ Y i
;; (i.e., the prediction that our regression line is making)
;; . Thirdly, I changed the letters used to describe the coefficients from
;; m
;; and
;; c
;; to
;; b
;; 1
;; and
;; b 0
;; . That’s just the way that statisticians like to refer to the coefficients in a regression model. I’ve no idea why they chose
;; b
;; , but that’s what they did. In any case
;; b 0
;; always refers to the intercept term, and
;; b 1
;; refers to the slope.

;; Excellent, excellent. Next, I can’t help but notice that – regardless of whether we’re talking about the good regression line or the bad one – the data don’t fall perfectly on the line. Or, to say it another way, the data
;; Y i
;; are not identical to the predictions of the regression model
;; ^ Y i
;; . Since statisticians love to attach letters, names and numbers to everything, let’s refer to the difference between the model prediction and that actual data point as a residual, and we’ll refer to it as
;; ϵ i
;; .215 Written using mathematics, the residuals are defined as:
;; ϵ i = Y i − ^ Y i

;; which in turn means that we can write down the complete linear regression model as:
;; Y i = b 1 X i + b 0 + ϵ i
;;
;; ## 15.2 Estimating a linear regression model

^kinds/vega
(hc/xform reg-plot/regression-chart
          :TITLE "The best fitting regression line "
          :REGRESSION-X :dan-sleep
          :REGRESSION-Y :dan-grump
          :REGRESSION-POINT-DATA (seq (ds/rows data :as-maps))
          :REGRESSION-LINE-DATA predictions)





(def residuals
  (-> predictions-ds
      (ds/add-column :residual
       (seq
        (.residuals
         (ml/thaw-model lm))))
      (ds/add-column :pred+resid
                     (fn [ds] (func/+ (:residual ds) (:dan-grump ds))))))

^kinds/vega
(hc/xform ht/point-chart
         :DATA (-> residuals (ds/rows :as-maps) seq)
         :X :dan-sleep :Y :pred+resid
         :MCOLOR :black
         :XTITLE "My sleep (hours)"
         :YTITLE "Residual"
         :XSCALE {"zero" false}
         :YSCALE {"zero" false})

^kinds/vega
(hc/xform (assoc ht/view-base
                 :mark (merge ht/mark-base {:type "rule"}))

         :DATA (seq (ds/rows residuals :as-maps))
         :ENCODING {:x {:field :dan-sleep :type :quantitative :scale {:zero false}}
                    :x2 {:field :dan-sleep  :type :quantitative}
                    :y  {:field :dan-grump  :type :quantitative :scale {:zero false}}
                    :y2 {:field :pred+resid  :type :quantitative}})

^kinds/vega
(hc/xform ht/layer-chart
         :HEIGHT 400 :WIDTH 450
         :LAYER
         [(hc/xform (assoc ht/view-base
                           :mark (merge ht/mark-base {:type "rule"}))

                    :DATA (seq (ds/rows residuals :as-maps))
                    :ENCODING {:x {:field :dan-sleep :type :quantitative :scale {:zero false}}
                               :x2 {:field :dan-sleep  :type :quantitative}
                               :y {:field :dan-grump  :type :quantitative :scale {:zero false}}
                               :y2 {:field :pred+resid  :type :quantitative}})
          (hc/xform reg-plot/regression-line-chart
                    :X :dan-sleep :Y :dan-grump
                    :DATA predictions)
          (hc/xform ht/point-chart
                    :DATA (-> residuals (ds/rows :as-maps) seq)
                    :X :dan-sleep :Y :pred+resid
                    :MCOLOR :black)])
