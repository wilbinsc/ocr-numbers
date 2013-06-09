import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File

class CharacterRaster(image: Array[Int]) {
  val data = image
  var classification: Option[Char] = None
  
  def classify(c: Char) = {
    this.classification = Some(c)
    this
  }
  
  def draw() = {
    val img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
    img.getRaster().setPixels(0, 0, 28, 28, data)
    ImageIO.write(img, "png", new File(classification.get + ".png"))
  }
}

object CharacterRaster {
  def fromCsvImageData(data: String): CharacterRaster = {
    // parse a 28x28 2d unsigned byte array
    new CharacterRaster(data.split(",").map(_.charAt(0).toInt))
  }

  def fromCsvClassifiedImageData(data: String) = {
    fromCsvImageData(data.substring(2)).classify(data.charAt(0))
  }

  def distance(c1: CharacterRaster, c2: CharacterRaster) = {
    c1.data.zip(c2.data.toList).map((x: (Int, Int)) => ((x._1 - x._2) * (x._1 - x._2)).asInstanceOf[Long]).sum
  }
}