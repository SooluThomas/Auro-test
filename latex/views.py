# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse

import wikipedia
import wolframalpha

from nltk.tokenize import PunktSentenceTokenizer
import nltk

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

def tokenStr():
	filename = "./Files/userinput.txt"
	filename1 = "./Files/dictionary.txt"
	user = input("Enter Data: ")

	with open (filename, "w") as fd:
		fd.write(user)

	fd = open(filename,"r")
	print(fd.read())
	fd.close()

	fd = open(filename, "r")
	fd1 = open(filename1, "r")
	train_text = fd.read()
	# sample_tect = fd1.read()


	custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

	tokenized = custom_sent_tokenizer.tokenize(train_text)

	def process_content():
		try:
			for i in tokenized[:5]:
				words = nltk.word_tokenize(i)
				tagged = nltk.pos_tag(words)
				print(tagged)
				with open("./Files/tokenized.txt", "w") as fd, open(filename) as fd1, open("./Files/comp.txt") as fd2:
					# myWords = set(line.split(',') for line in tagged)
					with open("./Files/newtokens.txt", "w") as f:
						for line in tagged:
							print(line)
							freq1 = line[1]
							f.write(line + "\n")
						for line1 in fd2:
							freq= line1[1]
							if freq in freq1:
								print(freq)
								# fd.write(myWords)
				fd.close()
				chunkGram = r"""Chunk: {<.*>+}}<VB.?|IN|DT|TO>+{"""
				chunkParser = nltk.RegexpParser(chunkGram)
				chunked = chunkParser.parse(tagged)
				print(chunked)

		except Exception as e:
			print(str(e))
		
	process_content()
	fd1.close()
	fd.close()

def index(request):
	answer = newFn(request.GET.get('q', ''))
	return HttpResponse(answer)

def training():
	#Initializing

	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	#Training

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	#Evaluating Model

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	if __name__ == '__training__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
		                    help='Directory for storing input data')
		FLAGS, unparsed = parser.parse_known_args()
		tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

expected_questions = {'what is your name':'Auro', 
	'who developed you':'Ginu, Greeshma and Soolu',
	'who developed you?':'Ginu, Greeshma and Soolu',
	'who created you':'Ginu, Greeshma and Soolu',
	'who created you?':'Ginu, Greeshma and Soolu',
	'when were you developed':'I am still under delopment', 
	'what is auro': 'I am a personal Digital Assistant developed by Ginu, Greeshma and Soolu', 
	'who are you':'I am Auro, the Personal Assistant',
	'do you know me': 'Very well..!! I am afraid to reply to that. I am Auro, the Personal Assistant',
	'do you know me?': 'No, I\'m afraid you have me at a disadvantage there. I am Auro, the Personal Assistant',
	'how old are you?': 'A few days ago',
	'how old are you': 'A few days ago',
	'what is your age?': 'A few days',
	'what is your age': 'A few days',
	'when was i born': '21st April 2017. My Project Review is still going on...',
	'me': 'Auro',
	'you': 'Auro',
	'who is Soolu Thomas': 'Soolu, Ginu and Greeshma Developed me!',
	'who is Greeshma': 'Soolu, Ginu and Greeshma Developed me!',
	'who is Ginu': 'Soolu, Ginu and Greeshma Developed me!'
	'What programing language was used to develop you?': 'Python',
	'Development': 'Python',
	'what are you': 'A Personal Digital Assistant',}


def newFn(question):
	try:
		input = question
		input = input.strip('').lower() 
		app_id = "587E79-677J3JTXE3"
		client = wolframalpha.Client(app_id)
		try:
			res_exp = expected_questions[input]
			if input not in res_exp :
				res = client.query(input)
				answer = next(res.results).text
				return answer
			else:
				answer = res_exp
				return answer
		except:
			res = client.query(input)
			answer = next(res.results).text
			return answer
	except:
		wikipedia.set_lang("en")
		try:
			return "Wikipedia says: " + wikipedia.summary(input, sentences=2)
		except wikipedia.exceptions.DisambiguationError as e:
			return "Ambiguos question! Be more specific"
		except wikipedia.exceptions.PageError as pe:
			return "I am sorry. I didn't get your question"
