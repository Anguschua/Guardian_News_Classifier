object classifier {
    def runClassifierJob(): Unit = {
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._
        import org.apache.spark.sql.types.{ArrayType, StringType}
        import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer}
        import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
        import scala.util.matching.Regex
        import scala.collection.mutable
        import java.io.PrintWriter
        import scala.collection.immutable.ListMap

        val spark = SparkSession.builder()
        .appName("Classifier")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .getOrCreate()

        def removeSymbols(words: Seq[String]): Seq[String] = {
        val regex = new Regex("[^a-zA-Z0-9]")
        words.map(word => regex.replaceAllIn(word, "")).filter(_.nonEmpty)
        }

        val dataPath = "filtered_data.csv"
        val df = spark.read.option("header", "true").option("inferSchema", "true").csv(dataPath)

        var selectedDF = df.select("Title", "Section")
        selectedDF = selectedDF.withColumn("Section", trim(lower(regexp_replace(col("Section"), "[^\\w\\s-]", ""))))

        val validSections = Seq("football", "sport", "tv-and-radio", "politics", "news", "uk-news", "world", "australia-news", "environment", "science", "commentisfree", "media", "lifeandstyle")
        selectedDF = selectedDF.filter(col("Section").isin(validSections: _*))

        val tokenizer = new Tokenizer().setInputCol("Title").setOutputCol("Words")
        val tokenizedDF = tokenizer.transform(selectedDF)

        val removeSymbolsUDF = udf(removeSymbols _)

        val cleanedDF = tokenizedDF.withColumn("CleanedWords", removeSymbolsUDF(col("Words")))

        val stopWordsRemover = new StopWordsRemover().setInputCol("CleanedWords").setOutputCol("FilteredWords")
        val finalDF = stopWordsRemover.transform(cleanedDF)

        val vectorizer = new CountVectorizer().setInputCol("FilteredWords").setOutputCol("Features").setVocabSize(10000).setMinDF(2)
        val cvModel = vectorizer.fit(finalDF)
        val vectorizedDF = cvModel.transform(finalDF)

        val labelIndexer = new StringIndexer().setInputCol("Section").setOutputCol("label")
        val indexedDF = labelIndexer.fit(vectorizedDF).transform(vectorizedDF)

        val Array(trainDF, testDF) = indexedDF.randomSplit(Array(0.8, 0.2), seed = 42)

        val labelCounts = indexedDF.groupBy("label").count().collect()
        val fractions = labelCounts.map(row => row.getDouble(0) -> Math.min(100.0 / row.getLong(1), 1.0)).toMap
        val sampledTestDF = testDF.stat.sampleBy("label", fractions, 42)

        val lr = new LogisticRegression().setFeaturesCol("Features").setLabelCol("label").setMaxIter(10)

        val nb = new NaiveBayes().setFeaturesCol("Features").setLabelCol("label")

        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

        val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).build()
        val crossvalLR = new CrossValidator().setEstimator(lr).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(3)
        val cvLRModel = crossvalLR.fit(trainDF)
        val lrPredictions = cvLRModel.transform(sampledTestDF)
        val trainingAccuracyLR = cvLRModel.avgMetrics(0)
        val testingAccuracyLR = evaluator.evaluate(lrPredictions)

        val crossvalNB = new CrossValidator().setEstimator(nb).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(3)
        val cvNBModel = crossvalNB.fit(trainDF)
        val nbPredictions = cvNBModel.transform(sampledTestDF)
        val trainingAccuracyNB = cvNBModel.avgMetrics(0)
        val testingAccuracyNB = evaluator.evaluate(nbPredictions)

        val lrCM = lrPredictions.groupBy("label", "prediction").count().collect()
        val numLabels = labelIndexer.fit(vectorizedDF).labels.length
        val lrMatrix = Array.ofDim[Double](numLabels, numLabels)
        lrCM.foreach { row =>
        val label = row.getDouble(0).toInt
        val prediction = row.getDouble(1).toInt
        lrMatrix(label)(prediction) = row.getLong(2).toDouble
        }

        val lrMatrixWriter = new PrintWriter("logistic_regression_confusion_matrix.csv")
        lrMatrixWriter.write(lrMatrix.map(_.mkString(",")).mkString("\n"))
        lrMatrixWriter.close()

        val nbCM = nbPredictions.groupBy("label", "prediction").count().collect()
        val nbMatrix = Array.ofDim[Double](numLabels, numLabels)

        nbCM.foreach { row =>
        val label = row.getDouble(0).toInt
        val prediction = row.getDouble(1).toInt
        nbMatrix(label)(prediction) = row.getLong(2).toDouble
        }

        val nbMatrixWriter = new PrintWriter("naive_bayes_confusion_matrix.csv")
        nbMatrixWriter.write(nbMatrix.map(_.mkString(",")).mkString("\n"))
        nbMatrixWriter.close()
        
        println(s"Logistic Regression Training Accuracy: $trainingAccuracyLR")
        println(s"Logistic Regression Testing Accuracy: $testingAccuracyLR")
        println(s"Naive Bayes Training Accuracy: $trainingAccuracyNB")
        println(s"Naive Bayes Testing Accuracy: $testingAccuracyNB")
        println("Confusion matrices have been saved as CSV files.")

        spark.stop()

    }
}