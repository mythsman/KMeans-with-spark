
import scala.Array.canBuildFrom
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.MyKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.rdd.RDD

object Cluster {
  val sc = new SparkContext(new SparkConf());

  var inputPath: String = "";
  var outputPath: String = "";
  var clusters: Int = 0;

  def predictDenseByMllib(initializationMode: String, similarityMode: String) {
    val data = sc.textFile(inputPath).filter(str => str.trim.split(" ").length != 2)
    val parsedData = data.map(s => Vectors.dense(s.split(" ").map(_.toDouble))).cache

    val model = new MyKMeans()
      .setInitializationMode(initializationMode)
      .setK(this.clusters)
      .setRuns(1)
      .setSimilarityMode(similarityMode)
      .run(parsedData)
    val res = parsedData.map(item => model.predict(item))
    res.saveAsTextFile(outputPath)
  }

  def predictSparseByMllib(initializationMode: String, similarityMode: String) {
    val data = sc.textFile(inputPath)
    val row = data.first.trim.split(" ")(1).toInt

    val parsedData = data.filter(str => str.trim.split(" ").length != 3)
      .map(s => {
        val splited = s.trim.split("\\s+")
        val value = splited.filter(str => str.contains(".")).map(_.toDouble)
        val idx = splited.filterNot(str => str.contains(".")).map(_.toInt)
        Vectors.sparse(row + 1, idx, value)
      }).cache

    val tfidf = (new IDF(minDocFreq = 2)).fit(parsedData).transform(parsedData)

    val model = new MyKMeans()
      .setInitializationMode(initializationMode)
      .setK(this.clusters)
      .setRuns(10)
      .setSimilarityMode(similarityMode)
      .run(tfidf)

    val res = tfidf.map(item => model.predict(item))

    res.saveAsTextFile(outputPath)
  }

  def main(args: Array[String]) {
    if (args.length < 3) {
      println("Invalid arguments.")
      println("The arguments are <input path> <output path> <cluster number>.")
      return
    }
    this.inputPath = args(0)
    this.outputPath = args(1)
    this.clusters = args(2).toInt
    val data = sc.textFile(inputPath)
    val title = data.first.trim.split("\\s+")
    if (title.length == 2) {
      predictDenseByMllib("k-means++", "COS")
    } else {
      predictSparseByMllib("random", "COS")
    }
    //    this.inputPath = "hdfs://localhost:9000/user/myths/data/tr11WithTitle"
    //    (0 until 20).foreach(i => {
    //      this.outputPath = "hdfs://localhost:9000/user/myths/output" + i
    //      predictSparseByMllib("random", "COS")
    //    })

  }
}