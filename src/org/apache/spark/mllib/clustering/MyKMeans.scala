package org.apache.spark.mllib.clustering

import scala.Array.canBuildFrom
import scala.Option.option2Iterable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.BLAS.scal
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

class MyKMeans private (
  private var k: Int,
  private var maxIterations: Int,
  private var runs: Int,
  private var initializationMode: String,
  private var similarityMode: String,
  private var initializationSteps: Int,
  private var epsilon: Double,
  private var seed: Long) extends Serializable {

  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  val COS = "COS"
  val EUCLI = "EUCLI"

  var scaleVector: Vector = Vectors.dense(0)
  var varianceVector: Vector = Vectors.dense(0)

  def this() = this(2, 20, 1, "k-means||", "COS", 5, 1e-8, Utils.random.nextLong())

  def getK: Int = k

  def setK(k: Int): this.type = {
    this.k = k
    this
  }
  def getRuns: Int = runs

  def setRuns(runs: Int): this.type = {
    this.runs = runs
    this
  }
  def getMaxIterations: Int = maxIterations
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  def getInitializationMode: String = initializationMode

  def setInitializationMode(initializationMode: String): this.type = {
    validateInitMode(initializationMode)
    this.initializationMode = initializationMode
    this
  }

  def getSimilarityMode: String = similarityMode

  def setSimilarityMode(similarityMode: String): this.type = {

    this.similarityMode = similarityMode
    this
  }

  def getInitializationSteps: Int = initializationSteps

  def setInitializationSteps(initializationSteps: Int): this.type = {
    this.initializationSteps = initializationSteps
    this
  }

  def getEpsilon: Double = epsilon

  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }

  def getSeed: Long = seed

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  def run(data: RDD[Vector]): KMeansModel = {

    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()
    val zippedData = data.zip(norms).map {
      case (v, norm) =>
        new VectorWithNorm(v, norm)
    }
    val model = runAlgorithm(zippedData)
    norms.unpersist()
    model
  }

  private def runAlgorithm(data: RDD[VectorWithNorm]): KMeansModel = {

    val sc = data.sparkContext

    val numRuns = runs

    val centers =
      if (initializationMode == RANDOM) {
        initRandom(data)
      } else {
        initMyKMeansParallel(data)
      }

    val active = Array.fill(numRuns)(true)
    val costs = Array.fill(numRuns)(0.0)

    var activeRuns = new ArrayBuffer[Int] ++ (0 until numRuns)
    var iteration = 0

    while (iteration < maxIterations && !activeRuns.isEmpty) {
      type WeightedPoint = (Vector, Long)
      def mergeContribs(x: WeightedPoint, y: WeightedPoint): WeightedPoint = {
        axpy(1.0, x._1, y._1)
        (y._1, x._2 + y._2)
      }

      val activeCenters = activeRuns.map(r => centers(r)).toArray
      val costAccums = activeRuns.map(_ => sc.accumulator(0.0))

      val bcActiveCenters = sc.broadcast(activeCenters)

      val totalContribs = data.mapPartitions { points =>
        val thisActiveCenters = bcActiveCenters.value
        val runs = thisActiveCenters.length
        val k = thisActiveCenters(0).length
        val dims = thisActiveCenters(0)(0).vector.size

        val sums = Array.fill(runs, k)(Vectors.zeros(dims))
        val counts = Array.fill(runs, k)(0L)

        var arrAcc = Array.fill(runs, k)(0.0)
        points.foreach { point =>
          (0 until runs).foreach { i =>
            val (bestCenter, cost) = findClosest(thisActiveCenters(i), point)
            arrAcc(i)(bestCenter) += cost
            val sum = sums(i)(bestCenter)
            axpy(1.0, point.vector, sum)
            counts(i)(bestCenter) += 1
          }
        }

        (0 until runs).foreach { i =>
          costAccums(i) += arrAcc(i).sum
        }

        val contribs = for (i <- 0 until runs; j <- 0 until k) yield {
          ((i, j), (sums(i)(j), counts(i)(j)))
        }
        contribs.iterator
      }.reduceByKey(mergeContribs).collectAsMap()

      bcActiveCenters.unpersist(blocking = false)

      for ((run, i) <- activeRuns.zipWithIndex) {
        var changed = false
        var j = 0
        while (j < k) {
          val (sum, count) = totalContribs((i, j))
          if (count != 0) {
            scal(1.0 / count, sum)
            val newCenter = new VectorWithNorm(sum)
            if (getSquaredDistance(newCenter, centers(run)(j)) > epsilon) {
              changed = true
            }
            centers(run)(j) = newCenter
          }
          j += 1
        }
        if (!changed) {
          active(run) = false
        }
        costs(run) = costAccums(i).value
      }

      activeRuns = activeRuns.filter(active(_))
      iteration += 1
    }

    val (minCost, bestRun) = costs.zipWithIndex.min
    new KMeansModel(centers(bestRun).map(_.vector))
  }

  private def initRandom(data: RDD[VectorWithNorm]): Array[Array[VectorWithNorm]] = {
    val sample = data.takeSample(true, runs * k, new XORShiftRandom(this.seed).nextInt()).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).map { v =>
      new VectorWithNorm(Vectors.dense(v.vector.toArray), v.norm)
    }.toArray)
  }

  private def initMyKMeansParallel(data: RDD[VectorWithNorm]): Array[Array[VectorWithNorm]] = {
    val centers = Array.tabulate(runs)(r => ArrayBuffer.empty[VectorWithNorm])
    var costs = data.map(_ => Array.fill(runs)(Double.PositiveInfinity))

    val seed = new XORShiftRandom(this.seed).nextInt()
    val sample = data.takeSample(true, runs, seed).toSeq
    val newCenters = Array.tabulate(runs)(r => ArrayBuffer(sample(r).toDense))

    /** Merges new centers to centers. */
    def mergeNewCenters(): Unit = {
      var r = 0
      while (r < runs) {
        centers(r) ++= newCenters(r)
        newCenters(r).clear()
        r += 1
      }
    }

    var step = 0
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      val preCosts = costs
      costs = data.zip(preCosts).map {
        case (point, cost) =>
          Array.tabulate(runs) { r =>
            math.min(pointCost(bcNewCenters.value(r), point), cost(r))
          }
      }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs
        .aggregate(new Array[Double](runs))(
          seqOp = (s, v) => {
            var r = 0
            while (r < runs) {
              s(r) += v(r)
              r += 1
            }
            s
          },
          combOp = (s0, s1) => {
            var r = 0
            while (r < runs) {
              s0(r) += s1(r)
              r += 1
            }
            s0
          })

      bcNewCenters.unpersist(blocking = false)
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointsWithCosts.flatMap {
          case (p, c) =>
            val rs = (0 until runs).filter { r =>
              rand.nextDouble() < 2.0 * c(r) * k / sumCosts(r)
            }
            if (rs.length > 0) Some(p, rs) else None
        }
      }.collect()
      mergeNewCenters()
      chosen.foreach {
        case (p, rs) =>
          rs.foreach(newCenters(_) += p.toDense)
      }
      step += 1
    }

    mergeNewCenters()
    costs.unpersist(blocking = false)

    val bcCenters = data.context.broadcast(centers)
    val weightMap = data.flatMap { p =>
      Iterator.tabulate(runs) { r =>
        ((r, findClosest(bcCenters.value(r), p)._1), 1.0)
      }
    }.reduceByKey(_ + _).collectAsMap()

    bcCenters.unpersist(blocking = false)

    val finalCenters = (0 until runs).par.map { r =>
      val myCenters = centers(r).toArray
      val myWeights = (0 until myCenters.length).map(i => weightMap.getOrElse((r, i), 0.0)).toArray
      LocalKMeans.kMeansPlusPlus(r, myCenters, myWeights, k, 30)
    }

    finalCenters.toArray
  }

  private[mllib] def findClosest(
    centers: TraversableOnce[VectorWithNorm],
    point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val distance: Double = getSquaredDistance(center, point)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  private[mllib] def pointCost(
    centers: TraversableOnce[VectorWithNorm],
    point: VectorWithNorm): Double =
    findClosest(centers, point)._2

  private[spark] def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case RANDOM => true
      case K_MEANS_PARALLEL => true
      case _ => false
    }
  }

  private def getSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    similarityMode match {
      case COS =>
        val dot = v1.vector.toBreeze.dot(v2.vector.toBreeze)
        val cosDist = 1 - dot / (v1.norm * v2.norm)
        cosDist
      case EUCLI =>
        MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
    }
  }
}

private[clustering] class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}

