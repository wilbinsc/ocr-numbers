import java.io.File
import java.io.PrintWriter

import scala.Array.canBuildFrom
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

	def knn(test: CharacterRaster, trainingSet: List[CharacterRaster], f: (CharacterRaster, CharacterRaster) => Double) = {
		val neighbors = trainingSet.par.map((train) => (train.classification.get, f(test, train))).toList.sortBy(_._2)
		val result = List(kfilter(neighbors, 3), kfilter(neighbors, 5), kfilter(neighbors, 10), kfilter(neighbors, 20))
		// take the majority vote
		val consensus = result.groupBy(identity).mapValues(_.size).toList.sortBy(-_._2).apply(0)._1
		consensus :: result
	}

	def kfilter(neighbors: List[(Char, Double)], k: Int): Char = {
		neighbors.slice(0, k).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2 * -1).apply(0)._1
	}

	def confidenceRating(count: Int, sampleSize: Int): Int = {
		if (count >= 0.8 * sampleSize) 3
		else if (count >= 0.6 * sampleSize) 2
		else 1
	}

	def precomputePca() {
		val trainingSet = loadTraining.toList
		val pca = new Pca
		pca.setup(trainingSet.size, 28 * 28)
		trainingSet.foreach((c) => pca.addSample(c.getDataAsDouble))
		List(10, 20, 30, 40, 50, 70, 100).foreach(num => {
			pca.computeBasis(num)
			val file = new PrintWriter(new File("pca" + num))
			trainingSet.foreach((c) => {
				file.write(c.classification.get)
				file.write(',')
				file.write(pca.transform(c.getDataAsDouble).mkString(","))
				file.write('\n')
			})
			file.close()
		})
	}

	def main(args: Array[String]): Unit = {
		precomputePca()
//		val trainingSet = loadTraining.toList
//		val pca = new Pca
//		pca.setup(trainingSet.size, 28 * 28)
//		trainingSet.foreach((c) => pca.addSample(c.getDataAsDouble))
//		pca.computeBasis(50)
//		trainingSet.par.foreach((c) => c.pca = pca.transform(c.getDataAsDouble))
//		println("PCA done")
//		// val testSet = loadTest.toList
//		// split the training set
//		val split = trainingSet.partition(_.data.hashCode() % 50 == 0)
//		// calculate the error rates for k = 1,3,5,10,20,50
//		val errors: Array[Int] = new Array[Int](5)
//		for (test <- split._1) {
//			val classification = knn(test, split._2, CharacterRaster.pcaDistance)
//			for (i <- 0 until classification.length; if classification(i) != test.classification.get)
//				errors(i) = errors(i) + 1
//		}
//		errors.map(_.asInstanceOf[Double] / split._1.size).toArray.foreach(println)
	}
}