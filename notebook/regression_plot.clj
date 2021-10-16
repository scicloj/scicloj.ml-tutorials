(ns regression-plot
  (:require

   [aerial.hanami.common :as hc]
   [aerial.hanami.templates :as ht]
   [aerial.hanami.core :as hmi]))


(def regression-line-chart
  (hc/xform ht/line-chart
            :DATA :VALDATA
            :X :X :Y :Y
            :MCOLOR :blue
            :XSCALE {"zero" false}
            :YSCALE {"zero" false}))

(def regression-points-chart
  (hc/xform ht/point-chart
            :TITLE :TITLE
            :DATA :VALDATA
            :X :X :Y :Y
            :MCOLOR :black
            :XSCALE {"zero" false}
            :YSCALE {"zero" false}))


(def regression-chart
  (hc/xform ht/layer-chart
            :TITLE :TITLE
            :HEIGHT 400 :WIDTH 450
            :X :REGRESSION-X
            :Y :REGRESSION-Y
            :LAYER
            [(hc/xform regression-points-chart
                       :DATA :REGRESSION-POINT-DATA
                       :X :REGRESSION-X
                       :Y :REGRESSION-Y)


             (hc/xform regression-line-chart
                       :DATA :REGRESSION-LINE-DATA
                       :X :REGRESSION-X
                       :Y :REGRESSION-Y)]))



            ;; :X :X
            ;; :Y :Y

(def residual-plot-chart
  (hc/xform ht/layer-chart
            :HEIGHT 400 :WIDTH 450

            :LAYER
            [(hc/xform (assoc ht/view-base
                              :mark (merge ht/mark-base {:type "rule"}))

                       :DATA :DATA-RESIDUALS
                       :ENCODING {:x {:field :X-REGRESSION :type :quantitative :scale {:zero false}}
                                  :x2 {:field :X-REGRESSION  :type :quantitative}
                                  :y {:field :Y-REGRESSION  :type :quantitative :scale {:zero false}}
                                  :y2 {:field :RESIDUAL+PREDICTION  :type :quantitative}})
             (hc/xform regression-line-chart
                     :X :X-REGRESSION :Y :Y-REGRESSION
                     :DATA :DATA-REGRESSION-LINE)
             (hc/xform ht/point-chart
                       :DATA :DATA-RESIDUALS
                     :X :X-REGRESSION :Y :Y-REGRESSION
                     :MCOLOR :black)]))


(def residual-plot-chart-2
  (hc/xform ht/layer-chart
            :HEIGHT 400 :WIDTH 450

            :LAYER
            [(hc/xform (assoc ht/view-base
                              :mark (merge ht/mark-base {:type "rule"}))

                       :DATA :DATA-RESIDUALS
                       :ENCODING {:x {:field :X-REGRESSION :type :quantitative :scale {:zero false}}
                                  :x2 {:field :X-REGRESSION  :type :quantitative}
                                  :y {:field :Y-REGRESSION  :type :quantitative :scale {:zero false}}
                                  :y2 {:field :RESIDUAL+PREDICTION  :type :quantitative}})
             (hc/xform ht/line-chart
                     :X :X-REGRESSION :Y :Y-REGRESSION
                     :DATA :DATA-REGRESSION-LINE)
             (hc/xform ht/point-chart
                       :DATA :DATA-RESIDUALS
                     :X :X-REGRESSION :Y :Y-REGRESSION
                     :MCOLOR :black)]))


(def residual-plot-chart-3
  (hc/xform ht/layer-chart
            :LAYER [(assoc ht/view-base
                           :mark (merge ht/mark-base {:type "rule" :color "green"})
                           :data {:values :DATA-RESIDUALS}
                           :encoding
                           (merge ht/encoding-base {:x {:field :X-REGRESSION :type :quantitative :scale {:zero false}}
                                                    :x2 {:field :X-REGRESSION  :type :quantitative}
                                                    :y {:field :Y-REGRESSION  :type :quantitative :scale {:zero false}}
                                                    :y2 {:field :RESIDUAL+PREDICTION  :type :quantitative}}))
                    (hc/xform ht/point-chart
                              :DATA :DATA-RESIDUALS
                              :X :X-REGRESSION
                              :Y :Y-REGRESSION)

                    
                    (hc/xform
                     (assoc  ht/line-layer
                             :data {:values :DATA-REGRESSION-LINE})
                     :X :X-REGRESSION
                     :Y :Y-REGRESSION)]))
