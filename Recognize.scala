import scala.io.Source

case class Digit(label: Int, pixels: List[Int])
case class Distance(label: Int, value: Double)

class Recognize{

  val training = loadFile("digitssample.csv")
  val samples = loadFile("digitscheck.csv")

  def loadFile(fileName: String): List[Digit] = {
    val lines = Source.fromFile(fileName).getLines.toList
    val digits = lines.tail.map(line => {val splitLine = line.split(","); new Digit(splitLine.head.toInt, splitLine.tail.map(_.toInt).toList)})
    digits
  }

  def distance(image1: List[Int], image2: List[Int]): Double = {
    Math.sqrt((image1, image2).zipped.toList.map(point => Math.pow(2, Math.abs(point._1 - point._2))).sum)
  }

  def OneNNClassify(sample: Digit): Int = {
    val distances = training.map(trainee => Distance(trainee.label, distance(sample.pixels, trainee.pixels)))
    val nearestNeighbors = distances.sortBy(distance => distance.value).head
    val classification = nearestNeighbors.value
    classification.toInt
  }

  def KNNClassify(sample: Digit, k: Int): Int = {
    val distances = training.map(trainee => Distance(trainee.label, distance(sample.pixels, trainee.pixels)))
    val closestNeighbor = distances.sortBy(distance => distance.value).head.value
    val weightedDistances = distances.map(distance => Distance(distance.label, distance.value / closestNeighbor))
    val nearestNeighbors = weightedDistances.sortBy(distance => distance.value).take(k)
    val classification = nearestNeighbors.groupBy(_.label).mapValues(_.size).maxBy(_._2)._1
    classification
  }

  def classifyAll(k: Int): List[Tuple2[Int, Int]] = {
    samples.map(sample =>  (sample.label, KNNClassify(sample, k)))
  }

  //List of (expected, actual) tuples.
  def benchmark(classification: List[Tuple2[Int, Int]]) = {
    classification.map(x => x._1 == x._2).filter(_ == true).size / samples.size.toFloat
  }
}
