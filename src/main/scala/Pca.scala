import org.ejml.data.DenseMatrix64F
import org.ejml.factory.DecompositionFactory
import org.ejml.ops.SingularOps
import org.ejml.ops.CommonOps

class Pca {

	var transformer: DenseMatrix64F = null
	var components: Int = -1
	var sample: DenseMatrix64F = new DenseMatrix64F(1, 1)
	var currSampleIndex: Int = 0
	var mean: Array[Double] = null

	def setup(numSamples: Int, vectorLen: Int) = {
		mean = new Array[Double](vectorLen)
		sample.reshape(numSamples, vectorLen, false)
		currSampleIndex = 0
	}

	def addSample(vector: Array[Double]) = {
		for (i <- 0 until vector.length) {
			sample.set(currSampleIndex, i, vector(i))
			mean(i) += vector(i)
		}
		currSampleIndex = currSampleIndex + 1
	}

	def computeBasis(numComponents: Int) = {
		components = numComponents;
		// compute vector mean
		for (i <- 0 until mean.length)
			mean(i) = mean(i) / currSampleIndex
		// subtract the mean from the original data
		for (i <- 0 until currSampleIndex; j <- 0 until mean.length)
			sample.set(i, j, sample.get(i, j) - mean(j))

		// Compute SVD and save time by not computing U
		val svd = DecompositionFactory.svd(sample.getNumRows(), sample.getNumCols(), false, true, false)
		svd.decompose(sample)
		transformer = svd.getV(null, true)
		val w = svd.getW(null)

		// Singular values are in an arbitrary order initially
		SingularOps.descendingOrder(null, false, w, transformer, true)

		// drop off extra components leaving components specified
		transformer.reshape(components, mean.length, true)
	}
	
	def transform(data: Array[Double]) = {
        val m = DenseMatrix64F.wrap(mean.length, 1, mean)
        val s = DenseMatrix64F.wrap(mean.length, 1, data)
        val r = new DenseMatrix64F(components, 1)
        CommonOps.sub(s, m, s)
        CommonOps.mult(transformer, s, r)
        r.data
	}
}