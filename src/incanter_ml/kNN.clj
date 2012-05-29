(ns incanter-ml.kNN
  (:use [incanter core stats datasets]))

(defn calc-distances
    "calcs distances between training-set and given point"
    [training-set X]
    (map #(euclidean-distance X %) (to-matrix training-set)))

(defn classify
    "kNN classifier
    For every point in given dataset:
        1. calculate the distance between in X and the current point
        2. sort the distances in increasing order
        3. take k items with lowest distances to in X
        4. find the majority class among these items
        5. return the majority class as our prediction for the class of in X
    "
    [k, training-set, labels, example]
    (let [distances  (calc-distances training-set example)
          labels (vector labels) ;remove headers to match with distances-vector
          labeled-distances  (dataset [:distances :labels] (map vector distances labels))]
            (->> labeled-distances
                ($order :distances :asc)
                ($ (range k) :all) ;;get top-k most similar object
                ($rollup :count :count-field :labels) ;;count labels
                ($order  :count-field :desc) ;; order labels by counts such as most popular is first
                ($ 0 :labels) ;;return most popular label
            )))

(defn test "" [test-data labels] (println "Deploy"))

(defn demo
    "Demo usage and testing functionality"
    []
    (let [iris (get-dataset :iris)
          training-set ($ :all [:not :Species] iris)
          labels  ($ :Species iris)
          example  ($ 9 :all training-set)
          k  (int 3)]
        (classify k training-set (vector labels) example)))