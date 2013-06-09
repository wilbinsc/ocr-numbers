import java.io.File
import java.io.PrintWriter

import scala.io.Source

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
		val neighbors = trainingSet.par.map((train) => (train.classification.get, CharacterRaster.distance(test, train))).toList.sortBy(_._2)
		val best = neighbors.slice(0, k).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1).apply(0)
		val best2 = neighbors.slice(0, 2 * k).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1).apply(0)
		val best5 = neighbors.slice(0, 5 * k).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1).apply(0)
		val best10 = neighbors.slice(0, 10 * k).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1).apply(0)
		// check if all model agree
		val allAgree = best._1 == best2._1 && best._1 == best5._1 && best._1 == best10._1
		// max confidence
		val maxConfidence = List(confidenceRating(best._2, k), confidenceRating(best2._2, 2 * k), confidenceRating(best5._2, 5 * k), confidenceRating(best10._2, 10 * k)).max
		(if (allAgree) "Y" else "N") + maxConfidence + ":" + best._1
	}

	def confidenceRating(count: Int, sampleSize: Int): Int = {
		if (count >= 0.8 * sampleSize) 3
		else if (count >= 0.6 * sampleSize) 2
		else 1
	}

	def main(args: Array[String]): Unit = {
		val trainingSet = loadTraining.toList
		// val testSet = loadTest.toList
		// split the training set
		val split = trainingSet.partition(_.data.hashCode() % 20 == 0)
		val file = new PrintWriter(new File("output.csv"))
		val k = 10
		for (test <- split._1) {
			val classification = knn(test, k, split._2)
			file.write(classification)
			if (classification.charAt(classification.length() - 1) != test.classification.get)
				file.write("e" + test.classification.get)
			file.write('\n')
			file.flush
		}
		file.close()
	}
}