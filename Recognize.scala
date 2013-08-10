import scala.io.Source

case class Digit(label: Int, pixels: List[Int])

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
    val distances = training.map(trainee => (trainee.label, distance(sample.pixels, trainee.pixels)))
    val nearestNeighbors = distances.sortBy(distance => distance._2).head
    val classification = nearestNeighbors._1
    classification
  }

  def KNNClassify(sample: Digit, k: Int): Int = {
    val distances = training.map(trainee => (trainee.label, distance(sample.pixels, trainee.pixels)))
    val nearestNeighbors = distances.sortBy(distance => distance._2).take(k)
    val classification = nearestNeighbors.groupBy(_._1).mapValues(_.size).maxBy(_._2)._1
    classification
  }

  def classifyAll(k: Int): List[Tuple2[Int, Int]] = {
    samples.map(sample =>  (sample.label, KNNClassify(sample, k)))
  }
}
