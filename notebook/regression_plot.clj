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
            :DATA :VALDATA
            :X :X :Y :Y
            :MCOLOR :black
            :XSCALE {"zero" false}
            :YSCALE {"zero" false}))


(def regression-chart
  (hc/xform ht/layer-chart
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
