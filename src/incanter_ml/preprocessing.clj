(ns solution.preprocessing
    (:use '(incanter core io charts stats infix)))

(def path "../")

(def training-file (str path "Psychopath_Trainingset_v1.csv"))
(def test-file (str path "Psychopath_Testset_v1.csv"))

(def training-data (read-data training-file {:header true}))

;;normalize columns

;;denormalize column


