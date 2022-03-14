import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from sklearn.model_selection import train_test_split

from pyspark_small import extract_label


def build(width, height, depth):
	inputs = Input((height, width, depth))
	s = Lambda(lambda x: x / 255) (inputs)

	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c2)

	model = Model(inputs=[inputs], outputs=[outputs])

	# return the constructed network architecture
	return model

if __name__ == '__main__':
	spark = SparkSession.builder.master("local[1]") \
		.appName('fruitrecognition') \
		.getOrCreate()

	# path to your image source directory
	img_dir = '/Users/jane/Documents/workspace/FruitRecognition/data/test/**'
	# Read image data using new image scheme
	image_df = spark.read.format("image").load(img_dir)

	selected_df = image_df.select("image.origin")

	new_df = selected_df.withColumn('label', selected_df.origin)

	# df.withColumn('year', split(df['dob'], '-').getItem(0))
	new_df = new_df.withColumn('label', extract_label(new_df['label']))

	new_df.show()
	# selected_df.withColumn("label", selected_df.origin.split('/')[-2])

	print('convert to pandas df')
	pd_df = new_df.toPandas()

	a = pd_df['origin'][0]
	print(a)
	pd_df['path'] = pd_df.apply(lambda row: row.origin.split('//')[1], axis=1)
	pd_df['label'].unique()

	train_set, test_set = train_test_split(pd_df, test_size=0.2, random_state=17)

	# Generating Spark Context
	X_train = train_set['origin']
	y_train = train_set['label']

	sc = spark.sparkContext
	# Build and compile U-Net model
	model = build(width=227, height=227,depth=3)
	opt = tf.optimizers.Adam()
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

	# Convert TrainX and TrainY to RDD
	rdd = to_simple_rdd(sc, X_train, y_train)

	print('create Spark model')
	# Create Spark model
	spark_model = SparkModel(model, frequency='epoch', mode='asynchronous',
	num_workers=1)
	print('fit on training rdd')
	spark_model.fit(rdd, epochs=1, batch_size=32, verbose=1, validation_split=0.1)
	print('done with fitting')
	# Evaluate Spark model by evaluating the underlying model
	# score = spark_model.master_network.evaluate(testX, testY, verbose=1)

	# print('Test accuracy:', score[1])
	# print(spark_model.get_results())