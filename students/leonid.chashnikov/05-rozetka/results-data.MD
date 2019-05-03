##2 classes distribution, 1-3 neg, 4-5 pos:

###Lemmatized text

#####with tone, count vectorizer:
GaussianNB:
	f1 0.5271, precision 0.5383, recall 0.5685
Perceptron:
	f1 0.7059, precision 0.7143, recall 0.7067
LogReg:
	f1 0.7271, precision 0.8043, recall 0.6901


#####no tone, count vectorizer:
GaussianNB:
	f1 0.5271, precision 0.5383, recall 0.5685
Perceptron:
	f1 0.7129, precision 0.7153, recall 0.7197
LogReg:
	f1 0.714, precision 0.7956, recall 0.6773



##2 classes distribution:

###Lemmatized text

#####with tone, tfidf vectorizer:
GaussianNB:
	f1 0.5549, precision 0.5503, recall 0.5878
Perceptron:
	f1 0.6757, precision 0.7434, recall 0.6522
LogReg:
	f1 0.5136, precision 0.8812, recall 0.5189


#####no tone, tfidf vectorizer:
GaussianNB:
	f1 0.5548, precision 0.5502, recall 0.5878
Perceptron:
	f1 0.6841, precision 0.84, recall 0.6364
LogReg:
	f1 0.4795, precision 0.5554, recall 0.5015


#####with tone, count vectorizer:
GaussianNB:
	f1 0.5549, precision 0.5503, recall 0.5883
Perceptron:
	f1 0.6997, precision 0.7099, recall 0.6987
LogReg:
	f1 0.6944, precision 0.8168, recall 0.6508


#####no tone, count vectorizer:
GaussianNB:
	f1 0.5549, precision 0.5503, recall 0.5883
Perceptron:
	f1 0.7015, precision 0.7328, recall 0.6819
LogReg:
	f1 0.674, precision 0.8242, recall 0.6302


##3 classes distribution:

###Not lemmatized text

#####with tone, tfidf vectorizer:
GaussianNB:
	f1 0.4219, precision 0.427, recall 0.4279
Perceptron:
	f1 0.4671, precision 0.5993, recall 0.4458
LogReg:
	f1 0.339, precision 0.6991, recall 0.3486


#####no tone, tfidf vectorizer:
GaussianNB:
	f1 0.4212, precision 0.4273, recall 0.4268
Perceptron:
	f1 0.4939, precision 0.7926, recall 0.445
LogReg:
	f1 0.3121, precision 0.4202, recall 0.3352


#####with tone, count vectorizer:
GaussianNB:
	f1 0.4242, precision 0.4281, recall 0.4299
Perceptron:
	f1 0.533, precision 0.6014, recall 0.5043
LogReg:
	f1 0.5161, precision 0.6477, recall 0.4739


#####no tone, count vectorizer:
GaussianNB:
	f1 0.4245, precision 0.4291, recall 0.4301
Perceptron:
	f1 0.5269, precision 0.6007, recall 0.4987
LogReg:
	f1 0.507, precision 0.662, recall 0.4628



###Lemmatized text

#####with tone, tfidf vectorizer:
GaussianNB:
	f1 0.411, precision 0.4135, recall 0.4325
Perceptron:
	f1 0.5045, precision 0.6169, recall 0.4977
LogReg:
	f1 0.3539, precision 0.6557, recall 0.3571


#####no tone, tfidf vectorizer:
GaussianNB:
	f1 0.4118, precision 0.4166, recall 0.4316
Perceptron:
	f1 0.5601, precision 0.7218, recall 0.5161
LogReg:
	f1 0.322, precision 0.5768, recall 0.34


#####with tone, count vectorizer:
GaussianNB:
	f1 0.4172, precision 0.4266, recall 0.4352
Perceptron:
	f1 0.5527, precision 0.5881, recall 0.5358
LogReg:
	f1 0.5394, precision 0.6628, recall 0.4964


#####no tone, count vectorizer:
GaussianNB:
	f1 0.4172, precision 0.4266, recall 0.4352
Perceptron:
	f1 0.5428, precision 0.5711, recall 0.5343
LogReg:
	f1 0.5345, precision 0.6683, recall 0.4892

