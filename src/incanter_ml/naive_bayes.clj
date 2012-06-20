(:ns incanter-ml.naive-bayes
     (:use '[incanter core])
     (:import '[java.util UUID]))

(defn join-feature-label
  "Maps feature with labels"
  [col-name feature label]
  (map vector (take (count feature) (repeat col-name)) feature label))

;;TODO rename as to-training-tasks or/and refactor it with to-classify-tasks
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
    (partition-all 5000) ;; split task-list to units
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

(defn calc-vs
  "Calcs V param of Laplace smoother for every features in freq-table and
  adds results as new column in freq-table.
  V-param is count of distinct values of features in training-set"
  [freq-table]
  (->>
    ($group-by :features freq-table)
    (map (fn [key-val]
           [(:features (first key-val))
            (count (set ($ :values (second key-val))))
            ]))
    (dataset [:features :vs])
    ;;add values as new column to freq-table
    ((fn [new-dt-set]
       ($join [:features :features] new-dt-set freq-table)))
    ))

(defn calc-ns
  "Calcs sum of different labels of features and adds values to given
  freq-table."
  [freq-table]
  (->>
    ($group-by [:features :labels] freq-table)
    (map (fn [key-val]
            (conj  
                (vals (first key-val))
                (sum ($ :freqs (second key-val)))) 
           ))
    (dataset [:ns :labels :features])
    ((fn [new-dt-set]
       ($join [[:features :labels] [:features :labels]] new-dt-set freq-table)))
    ))

(defn calc-laplace
  "Calculates laplace smoothing for non-existing features
  Input parameters:
    alpha - smoother coefficent
    V     - count of distinct features in training-set ~ size of dictionary
    N     - size of training-set 
    xi    - frequency of feature
  "
  [alpha V N xi]
  (/ 
      ($= xi + alpha * 1.0)
      ($= N + alpha * V + 1.0)))


(defn calc-likelihoods
  "Calculates likelihoods for given pair of evidence and hypothesis
  by using Laplace smoothing: ($= (xi * alpha)/(N + alpha * V))"
  [freq-table alpha]
  (->> freq-table
    ;;calc smoother parameters:
    (calc-vs)
    (calc-ns)
    ;;calc likelihoods
    ($map (fn [xi N V] (calc-laplace alpha V N xi)) [:freqs :ns :vs])
    ;;append results to table
    (dataset [:likelihoods])
    (conj-cols freq-table)  
    ))

(defn freq-table->likelihoods
  "Calculates likelihoods for given table of frequency, 
  and  recalculating priors"
  [freq-table alpha]
  (calc-likelihoods freq-table alpha))

;;TODO: refactor it to core-module
;;TODO: add support to params diction
(defn build-model
  "Returns hash-map with model description and important values"
  [freq-table alpha-coef]
  (let [updated-freq-table (->> 
                             freq-table
                             (freq-table->priors)
                             (freq-table->likelihoods))]
    (hash-map
      :id :naive-bayes
      :description "Naive Bayes with Laplace smoothing. 
                   This version accepts only discrete features and discerete labels.
                   For continous features use GaussianNaiveBayes or use binning"
      :feature-types [:discrete]
      :label-types [:dicrete]
      :alpha alpha-coef
      :freq-table updated-freq-table
      )    
    ))

(defn train
  "Trains model. alpha-coef is coefficent of Laplace smoother"
  [alpha-coef features labels]
  (->>
    (build-task-list features labels)
    (process-task-list)
    (map->freqtable)
    (build-model alpha-coef)
    )) 
 
;; -- CLASSIFY part -----------------------------------------------------------
;; TODO: end classifier and reduce reduntancy
;; TODO: refactor to utils or core module

(defn random-uuid
  "Clojure wrapper around java UUID/randomUUID"
  []
  (str (UUID/randomUUID)))



;TODO: what if continuous value ??
;TODO: likelihoods vs log-likelihoods ?? 
(defn get-likelihood
  "Gets likelihood from frequency tables if it exists.
  Or else returns value from laplace smoother"
  [model label feature value]
  (let [freq-table (:freq-table model)
        alpha (:alpha model)
        row ($where {:labels label
                    :features feature
                    :values value} freq-table)]
      (if (= 0 (first (dim row)))
          ;;if there's no attribute in table of likelihoods, then use Laplace smoothing
          (calc-laplace alpha
                       ($ :vs row);|V|
                       ($ :ns row);|N|
                        0;|xi|
                        )
          ;;else return value of :likelihoods-field
          ($ [:likelihoods] row))))

;TODO: replace static input params and model with partial func?
(defn row->tasks
  "Composes map/reducable tasks from each data row with  every classifying label.
  Usage:
    >>> (row-tasks model [:minor :major] [:hair :eye] ['blond' 'blue' 1])"
  [model labels row-map]
  (let [task-id (:task-ids row-map)
        feat-pairs  (dissoc row-map :task-ids)
        ]
    (for [label labels]
        (vector task-id 
                label
                (map (fn [feat-val]
                        (get-likelihood model label 
                            (first feat-val) (second feat-val))                       
                       ) feat-pairs)
        )
      )))

;;
;;TODO: bug in row-mapper
(defn to-classify-tasks
  "Builds lazy-seq of features posteriors"
  [model classify-set]
  (let [labels ($ :labels (:priors model))
        headers (col-names classify-set)
        row-mapper (partial row->tasks model labels)]
    (->>
      ;;generate row-id  and use those values as task-id and later as joining point for results 
      (:rows classify-set)
      (map (fn [row-map] (row-mapper row-map)))
      (reduce into [])
    )))

(defn add-task-ids
  "Adds task-id column to dataset to enable result joining"
  [dt-set]
  (conj-cols dt-set
        (dataset [:task-ids]
              (for [n (range 1 (first (dim dt-set)))] (random-uuid)))))

;;TODO: finish it
(defn classify
  ""
  [model classify-set]
  (let [data-coll (add-task-ids classify-set)]
    (->>
      (to-classify-tasks model data-coll)
      (partition-all 5000)
      ;;calc log-likelihoods for every combination of <task-id, label>
      (pmap 
        (fn [partition]
          (map
              (fn [task] 
                (vector 
                      (first task) ;get task-id
                      (second task) ; get label
                      (sum (log (last task)))) ;calc log-likelihoods for this class
              ) partition    
           )))
      ;;combine partitions to one vector 
      (reduce (fn [coll val]
                (into coll val)) [])

      ;;combine sub-result to final
      (reduce 
        (fn [sub-results task]
          (let [task-id (first task)
                vals (rest task)
                map-entry (find sub-results task-id)]
            (if (nil? map-entry)
              (assoc sub-results task-id (conj [] vals))
              (assoc sub-results task-id (conj (second map-entry) vals)))
            )) {})
      ;;sorts combined values to descending order
      (map
        (fn [map-entry]
          (let [task-id (first map-entry)
                vals (seq (second map-entry))]
            (vector task-id (sort-by  second > vals))
            )))
      ;;calc proibabilities 
      (map
        (fn [item]
          (let [task-id (first item)
               probs (second item)
               winner (first probs)
               winner-label (first winner)
               winner-prob (second winner)]
           (vector task-id 
                   winner-label
                   (exp (- winner-prob 
                           (sum (map #(second %1) probs)))))
          )))
      ;; prepare output data 
      (dataset [:rtask-ids :labels :probs])
      ($join [:task-ids :rtask-ids] data-coll) ;join prediction results with data-coll by task-ids
      ($ [:not :rtask-ids])
     )))

(defn test
  ""
  [model test-set]
  (println "TODO"))

(defn transform-demo-labels
    "Splits labels into 2 categories - minor, major"
    [labels]
    (map (fn [lbl]
              (if (< lbl 20) "minor" "major")) 
          labels))


;;Usage example 
(defn example-training
  "Sample usage based on dataset :hair-eye-color"
  []
  (let [training-set (get-dataset :hair-eye-color)
        features ($ [:not :count] training-set)
        labels (transform-demo-labels ($ :count training-set))
        alpha-coef 1.0]
    (train alpha-coef features labels)
    ))

(def  model (example-training))
(def test-set ($ [:not :count] (get-dataset :hair-eye-color)))
(classify model test-set)

