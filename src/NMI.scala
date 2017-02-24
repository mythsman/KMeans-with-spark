
import scala.Array.canBuildFrom
import com.google.common.math.DoubleMath
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.mutable.ArrayBuffer

object NMI {
  val sc = new SparkContext(new SparkConf());
  private def labelChecker(labelsTrue: Array[Int], labelsPred: Array[Int]): Unit = {
    require(labelsTrue.length == labelsPred.length && labelsTrue.length >= 2, "The length must be equal!" +
      "The size of labels must be greater than 1!")
  }

  private def mutualInformation(labelsTrue: Array[Int], labelsPred: Array[Int]) = {
    labelChecker(labelsTrue, labelsPred)
    val len: Int = labelsTrue.length
    val mapTrue: Map[Int, Int] = labelsTrue.groupBy(x => x).mapValues(_.length)
    val mapPred: Map[Int, Int] = labelsPred.groupBy(x => x).mapValues(_.length)
    labelsTrue.zip(labelsPred).groupBy(x => x).mapValues(_.length).map {
      case ((x, y), z) =>
        val wk = mapTrue(x)
        val cj = mapPred(y)
        val common = z.toDouble
        common * DoubleMath.log2(len * common / (1.0 * wk * cj)) / len
    }.sum
  }

  private def entropy(labels: Array[Int]) = {
    val N: Int = labels.length
    val array: Array[Int] = labels.groupBy(x => x).values.map(_.length).toArray
    array.map(x => -1.0 * x / N * DoubleMath.log2(1.0 * x / N)).sum
  }

  def normalizedMutualInformation(labelsTrue: Array[Int], labelsPred: Array[Int]): Double = {
    labelChecker(labelsTrue, labelsPred)
    mutualInformation(labelsTrue, labelsPred) / Math.sqrt(entropy(labelsTrue) * entropy(labelsPred))
  }

  def getNMI(path1: String, path2: String): Double = {
    val file1 = sc.textFile(path1)
    val file2 = sc.textFile(path2)
    val intArray1 = file1.toArray.map(res => res.toInt)
    val intArray2 = file2.toArray.map(res => res.toInt)
    NMI.normalizedMutualInformation(intArray1, intArray2)
  }
  def main(args: Array[String]) {
    println(getNMI(args(0), args(1)))
//    val file1 = "hdfs://localhost:9000/user/myths/data/tr11_Label"
//    var arr = ArrayBuffer[Double]()
//    (0 until 20).foreach(i => {
//      val file2 = "hdfs://localhost:9000/user/myths/output" + i
//      arr.append(getNMI(file1, file2))
//    })
//    arr.foreach(println)
//    println(arr.sum / arr.length)
  }
}