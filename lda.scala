object lda {
  def runLDAJob(): Unit = {
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types.{ArrayType, StringType}
    import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, CountVectorizer}
    import org.apache.spark.ml.clustering.LDA
    import scala.util.matching.Regex
    import scala.collection.immutable.ListMap
    import java.io.PrintWriter

    val spark = SparkSession.builder()
      .appName("lda")
      .config("spark.sql.shuffle.partitions", "300")
      .config("spark.executor.memory", "8g")
      .config("spark.driver.memory", "8g")
      .getOrCreate()

    import spark.implicits._

    def removeSymbols(words: Seq[String]): Seq[String] = {
      val regex = new Regex("[^a-zA-Z0-9]")
      words.map(word => regex.replaceAllIn(word, "")).filter(_.nonEmpty)
    }

    val dataPath = "dataV2.csv"
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

    val lda = new LDA().setK(5).setMaxIter(100).setFeaturesCol("Features")
    val ldaModel = lda.fit(vectorizedDF)

    val vocab = cvModel.vocabulary

    val topics = ldaModel.describeTopics(maxTermsPerTopic = vocab.length)

    val termProbabilities = Array.fill(vocab.length)(0.0)
    topics.collect().foreach { row =>
      val termIndices = row.getAs[Seq[Int]]("termIndices")
      val termWeights = row.getAs[Seq[Double]]("termWeights")
      termIndices.zip(termWeights).foreach { case (idx, prob) =>
        termProbabilities(idx) += prob
      }
    }

    val termsWithProbs = vocab.zip(termProbabilities)
    val top100Keywords = termsWithProbs.sortBy(-_._2).take(100)

    val sortedKeywordsMap = ListMap(top100Keywords.sortBy(-_._2): _*)
    val json = sortedKeywordsMap.map { case (term, prob) => "\"" + term + "\": " + prob }.mkString("{", ", ", "}")

    val writer = new PrintWriter("top_100_keywords.json")
    writer.write(json)
    writer.close()

    println("JSON file 'top_100_keywords.json' has been created successfully.")

    spark.stop()
    }
}
