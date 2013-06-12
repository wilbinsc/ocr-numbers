import java.io.File
import java.io.PrintWriter

import scala.Array.canBuildFrom
import scala.io.Source

object Main {

	val pcaN = List(40, 50, 60, 70, 80, 90, 100, 120, 140, 160)
	def loadTraining() = {
		for (line <- Source.fromFile("train.csv").getLines(); if (Character.isDigit(line.charAt(0))))
			yield CharacterRaster.fromCsvClassifiedImageData(line)
	}

	def loadTest() = {
		for (line <- Source.fromFile("test.csv").getLines(); if (Character.isDigit(line.charAt(0))))
			yield CharacterRaster.fromCsvImageData(line)
	}

	def loadTrainingPca(n: Int) = {
		for (line <- Source.fromFile("pca" + n).getLines())
			yield CharacterRaster.fromCsvClassifiedPcaData(line)
	}

	def loadTestPca(n: Int) = {
		for (line <- Source.fromFile("testpca" + n).getLines())
			yield CharacterRaster.fromCsvPcaData(line)
	}

	def knn(test: CharacterRaster, trainingSet: List[CharacterRaster], f: (CharacterRaster, CharacterRaster) => Double) = {
		val neighbors = trainingSet.par.map((train) => (train.classification.get, f(test, train))).toList.sortBy(_._2)
		List(
			kfilter(neighbors, 5, true),
			kfilter(neighbors, 5, false),
			kfilter(neighbors, 7, true),
			kfilter(neighbors, 7, false),
			kfilter(neighbors, 10, true),
			kfilter(neighbors, 10, false),
			kfilter(neighbors, 15, true),
			kfilter(neighbors, 15, false))
	}

	def knn(k: Int, test: CharacterRaster, trainingSet: List[CharacterRaster], f: (CharacterRaster, CharacterRaster) => Double) = {
		val neighbors = trainingSet.par.map((train) => (train.classification.get, f(test, train))).toList.sortBy(_._2)
		kfilter(neighbors, k, false)
	}

	def kfilter(neighbors: List[(Char, Double)], k: Int, uniformWeight: Boolean): Char = {
		if (uniformWeight)
			neighbors.slice(0, k)
				.groupBy(_._1)
				.mapValues(_.size)
				.toList
				.sortBy(_._2 * -1).apply(0)._1
		else
			neighbors.slice(0, k)
				.map(pair => (pair._1, 1 / Math.sqrt(pair._2))) // map the label, distance pair to label 1/sqrt(distance) as an inverse weight
				.groupBy(_._1)
				.mapValues(_.map(_._2).sum)
				.toList
				.sortBy(_._2 * -1).apply(0)._1
	}

	def confidenceRating(count: Int, sampleSize: Int): Int = {
		if (count >= 0.8 * sampleSize) 3
		else if (count >= 0.6 * sampleSize) 2
		else 1
	}

	def buildPca(samples: List[CharacterRaster]) = {
		val pca = new Pca
		pca.setup(samples.size, 28 * 28)
		samples.foreach((c) => pca.addSample(c.getDataAsDouble))
		pca
	}

	def precomputePca(pca: Pca, n: Int, data: List[CharacterRaster], outfile: String) {
		pca.computeBasis(n)
		val file = new PrintWriter(new File(outfile))
		data.foreach((c) => {
			if (c.classification != None) {
				file.write(c.classification.get)
				file.write(',')
			}
			file.write(pca.transform(c.getDataAsDouble).mkString(","))
			file.write('\n')
		})
		file.close()
	}

	def crossValidate(): Unit = {
		// laod the precomputed pca data
		val data = loadTraining.toList
		// split the training data into training and test sets, use 2% of training set for cross validation
		val (testSet, trainingSet) = data.partition(_.hashCode() % 50 == 0)
		// run the classification and output error rates
		var error = new Array[Double](8)
		for (
			test <- testSet;
			classification = knn(test, trainingSet, CharacterRaster.distance)
		) {
			for (i <- 0 until error.length; if classification(i) != test.classification.get)
				error(i) = error(i) + 1
		}
		println(error.map(_ / testSet.size).toList)
	}

	def classify() = {
		val trainingSet = loadTraining.toList
		val testSet = loadTest.toList
		val file = new PrintWriter(new File("weightedknn7.csv"))
		for (test <- testSet) {
			val classification = knn(7, test, trainingSet, CharacterRaster.distance)
			file.print(classification)
			file.print('\n')
			file.flush
		}
		file.close()
	}

	def main(args: Array[String]): Unit = {
		//		val trainingSet = loadTraining.toList
		//		val testSet = loadTest.toList
		//		val pca = buildPca(trainingSet ++ testSet)
		//		precomputePca(pca, 50, trainingSet, "pca50")
		//		precomputePca(pca, 50, testSet, "testpca50")
		classify()
		// loadTest.toList.apply(12).draw()
		// for (i <- 0 until 10)
		//	crossValidate
	}
}