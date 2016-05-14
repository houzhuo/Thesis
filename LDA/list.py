import numpy
ntd = numpy.array([(1,2,3),(4,5,6)],dtype='int32')
print ntd[:,1]
#p_t = numpy.divide(numpy.multiply(ntd[:, 1], nwt[word, :]), nt)
#t = numpy.random.multinomial(1, p_t / p_t.sum()).argmax()