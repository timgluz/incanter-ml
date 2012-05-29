(:ns incanter-ml.naive-bayes
     (:use '[incanter core]))

(defn join-feature-label
  "Maps feature with labels"
  [col-name feature label]
  (map vector (take (count feature) (repeat col-name)) feature label))
  
(defn build-task-list
  "Counts features and class pairs"
  [features labels]
  (let [job-list []
        col-names (col-names features)]
    (->> 
      (map (fn [col-name]
         (join-feature-label col-name 
                             ($ col-name features) labels)) col-names)
      ;; concanate task-list into same list
      (map #(into job-list %1))
      (reduce into [])
      )
    ))

(defn process-task-list
  ""
  [task-list]
  (->> task-list 
    (partition-all 500) ;; split task-list to units
    (pmap #(frequencies %1)) ;; process each tasks parallel
    (merge-with +) ;; concanate results-map into one map 
    (first)
    ))

(defn map->freqtable
  "Transforms map of counts to table of feature frequencies"
  [lazy-freq-map]
  (let [freq-map (doall lazy-freq-map)
        feat-vec (dataset [:features :values :labels] (keys freq-map))
        count-vec (dataset [:freqs] (vals freq-map))]
        (conj-cols feat-vec count-vec)
  ))

(defn split-by
  "splits given dataset to smaller datasets 
  by given values of features(column) to separate datasets "
  [col-name dt-set]
  (vals ($group-by col-name dt-set)))


(defn rows->dataset
  "Converts list of rows to dataset by makeing assumption,
  that headers are on first row. "
  [rows]
  (dataset (first rows) (rest rows)))

;;TODO: add support to params dictionary
(defn build-model
  "Returns hash-map with model description and important values"
  [freq-table]
  (hash-map :name "NaiveBayes" ;;(get params "NaiveBayes")
            :description "Simple naiveBayes"
            :type "classification"
            :freq-table freq-table
            :priors (freq-table->priors freq-table)
            :likelihoods (freq-table->likelihoods freq-table)
            ))


(defn train
  ""
  [features labels]
  (->>
    (build-task-list features labels)
    (process-task-list)
    (map->freqtable)
    (build-models freq-table)
    ))

(defn calc-priors
  "Calculates prior probabilities of freq-table by giving column.
  NB! Dont give all freq-table, it sums then all over all features."
  [col-name freq-table]
  (let [lbl-freq ($rollup :sum :freqs col-name freq-table)
        total-sum (sum ($ :freqs lbl-freq))]
      (conj-cols lbl-freq
            (dataset [:priors]
                     ($map #(/ %1 total-sum) [:freqs] lbl-freq)))))

(defn coll->dataset
  "Converts collection of vectors to dataset, 
  doing big assumptin that first row includes headers"
  [coll]
  (dataset (first coll) (rest coll)))

(defn freq-table->priors
  "Calculates prior probabilities for given frequency table by :labels column"
  [freq-table]
  (->>
    (split-by :features freq-table)
    ;;avoid double-readings
    (first)
    (calc-priors :labels)
    ))

(defn calc-likelihoods
  "Calculates likelihoods for given pair of evidence and hypothesis."
  [freq-table lbl-freq]
  (->>
    freq-table
    ;;($ [:freqs :labels] freq-table)
    ($map (fn [val lbl]
            (/ (* 1.0 val) 
               ($ :freqs ($where {:labels lbl} lbl-freq)))) [:freqs :labels])
    (dataset [:likelihoods])
    (conj-cols freq-table)  
    ))

(defn freq-table->likelihoods
  "Calculates likelihoods for given table of frequency, 
  and  recalculating priors"
  [freq-table]
  (let [priors (freq-table->priors freq-table)]
    (calc-likelihoods freq-table priors)))

;;TODO: end this
(defn classify
  ""
  [model & examples]
  (let [freq-table (get :freq-table model)
        priors (get :priors model)
        likelihoods (get :likelihoods model)]))

(defn test
  ""
  [model test-set])

(defn transform-demo-labels
    "Splits labels into 2 categories - minor, major"
    [labels]
    (map (fn [lbl]
              (if (< lbl 20) "minor" "major")) labels))

(defn example-usage
  "Sample usage based on dataset :hair-eye-color"
  []
  (let [training-set (get-dataset :hair-eye-color)
        features ($ [:not :count] training-set)
        labels (transform-demo-labels ($ :count training-set))]
    (train features labels)
    ))

(example-usage)

