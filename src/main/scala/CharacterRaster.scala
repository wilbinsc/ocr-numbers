import java.awt.image.BufferedImage
import java.io.File

import scala.Array.canBuildFrom

import javax.imageio.ImageIO

class CharacterRaster(image: Array[Int]) {
	val data = image
	var pca: Array[Double] = null
	var classification: Option[Char] = None

	def classify(c: Char) = {
		this.classification = Some(c)
		this
	}

	def getDataAsDouble() = {
		data.map(_.asInstanceOf[Double])
	}

	def draw() = {
		val img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
		img.getRaster().setPixels(0, 0, 28, 28, data)
		ImageIO.write(img, "png", new File((if (classification != None) classification.get else "x") + ".png"))
	}
}

object CharacterRaster {
	def fromCsvImageData(data: String): CharacterRaster = {
		// parse a 28x28 2d unsigned byte array
		new CharacterRaster(data.split(",").map(_.toInt))
	}

	def fromCsvPcaData(data: String): CharacterRaster = {
		val pca = data.split(",").map(_.toDouble)
		val c = new CharacterRaster(null)
		c.pca = pca
		c
	}

	def fromCsvClassifiedImageData(data: String) = {
		fromCsvImageData(data.substring(2)).classify(data.charAt(0))
	}

	def fromCsvClassifiedPcaData(data: String) = {
		fromCsvPcaData(data.substring(2)).classify(data.charAt(0))
	}

	def distance(c1: CharacterRaster, c2: CharacterRaster) = {
		c1.data.zip(c2.data.toList).map((x: (Int, Int)) => ((x._1 - x._2) * (x._1 - x._2)).asInstanceOf[Double]).sum
	}

	def pcaDistance(c1: CharacterRaster, c2: CharacterRaster) = c1.pca.zip(c2.pca.toList).map((x: (Double, Double)) => ((x._1 - x._2) * (x._1 - x._2)).asInstanceOf[Double]).sum
}