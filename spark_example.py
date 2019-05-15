
# 1. Loading the dataset
from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

df = spark.read.options(header="True", inferschema="True").csv("Admission.csv")
df.cache()

print("Total no. of rows: %d" % df.count())



# 2. Converting dataset to a dense vecctor
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
transformed_df.cache()
print(transformed_df.first())



# 3. Split train and test data
splits = [0.7, 0.3]
training_data, test_data = transformed_df.randomSplit(splits, 300)
print(training_data.first())

print("Number of training set rows: %d" % training_data.count())
print("Number of test set rows: %d" % test_data.count())



# 4. Training the Random Forest
from pyspark.mllib.tree import RandomForest

model = RandomForest.trainClassifier(training_data, numClasses=2, \
        numTrees=3, impurity="gini", \
        categoricalFeaturesInfo={}, maxDepth=4, seed=300) 



# 5. Predict and Calculate accuracy
predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (acc * 100))