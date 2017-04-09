
import scala.Array.canBuildFrom
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.MyKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.feature.Normalizer

object Cluster {
  val sc = new SparkContext(new SparkConf());

  var inputPath: String = "";
  var outputPath: String = "";
  var clusters: Int = 0;

  def predictDenseByMllib(initializationMode: String, similarityMode: String) {
    val data = sc.textFile(inputPath).map(s => Vectors.dense(s.split(" ").map(_.toDouble))).cache
    val model = new MyKMeans()
      .setInitializationMode(initializationMode)
      .setK(this.clusters)
      .setRuns(1)
      .setSimilarityMode(similarityMode)
      .run(data)
    val res = data.map(item => model.predict(item))
    res.saveAsTextFile(outputPath)
  }

  def predictSparseByMllib(initializationMode: String, similarityMode: String, row: Int) {

    val data = sc.textFile(inputPath)
      .map(s => {
        val splited = s.trim.split("\\s+")
        val value = new Array[Double](splited.size / 2)
        val idx = new Array[Int](splited.size / 2)
        for (i <- 0 to splited.size - 1) {
          if ((i & 1) != 0) {
            value.update(i / 2, splited.apply(i).toDouble)
          } else {
            idx.update(i / 2, splited.apply(i).toInt)
          }
        }
        Vectors.sparse(row + 1, idx, value)
      })
    val idf = new IDF(minDocFreq = 2).fit(data)
    val tfidf = idf.transform(data)
    val normalizer = new Normalizer()
    val normalized = tfidf.map(x => normalizer.transform(x)).cache
    val model = new MyKMeans()
      .setInitializationMode(initializationMode)
      .setK(this.clusters)
      .setRuns(10)
      .setSimilarityMode(similarityMode)
      .run(normalized)

    val res = normalized.map(item => model.predict(item))
    res.saveAsTextFile(outputPath)
  }

  def main(args: Array[String]) {
    if (args.length < 2) {
      println("Invalid arguments.")
      println("The arguments are <input path> <output path>.")
      return
    }
    this.inputPath = args(0)
    this.outputPath = args(1)
    //    this.clusters = 23
    //    predictDenseByMllib("k-means||", "COS")
    this.clusters = 20
    predictSparseByMllib("k-means||", "COS", 287985)

//    (0 until 20).foreach(i => {
//      this.outputPath = args(1) + i
//      predictSparseByMllib("k-means||", "COS", 287985)
//    })

  }
}
