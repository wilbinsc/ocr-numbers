import scala.io.Source
import java.io.PrintWriter
import java.io.File

object Main {

  def loadTraining() = {
    for (line <- Source.fromFile("train.csv").getLines(); if (Character.isDigit(line.charAt(0))))
      yield CharacterRaster.fromCsvClassifiedImageData(line)
  }

  def loadTest() = {
    for (line <- Source.fromFile("test.csv").getLines(); if (Character.isDigit(line.charAt(0))))
      yield CharacterRaster.fromCsvImageData(line)
  }
  
  def knn(test: CharacterRaster, k: Int, trainingSet: List[CharacterRaster]) = {
    val neighbors = trainingSet.par.map((train) => (train.classification.get, CharacterRaster.distance(test, train))).toList.sortBy(_._2).slice(0, k)
    val sortedHist = neighbors.groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1)
    sortedHist(0)
  }

  def main(args: Array[String]): Unit = {
    val trainingSet = loadTraining.toList
    val testSet = loadTest.toList
    val file = new PrintWriter(new File("output.csv"))
    val k = 10
    for (test <- testSet) {
      val classification = knn(test, k, trainingSet)
      val certainty = classification._2 match {
        case x if (9 to 10)contains(x) => '!'
        case y if (6 to 8)contains(y) => '*'
        case _ => '?'
      }
      file.write(certainty)
      file.write(classification._1)
      file.write('\n')
      file.flush
    }
    file.close()
  }
}