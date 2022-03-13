import csv
import numpy as np
from os import listdir
import tensorflow as tf
import tensorflowjs as tfjs

# A list of the files (with paths thereto) which will be used as data sources to train the model. 
# These files are not included in the present repository for copyright reasons; I used www.football-data.co.uk for this purpose
# (but had to modify their CSV files since the columns for the goals scored were not uniform in them)
list_of_files = listdir("data")

set_of_club_names = set()
number_of_games = 0

# A callback to end training when the (admittedly mediocre) accuracy of 54 % is reached
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if logs.get('accuracy') >= 0.54:
			print("\nEnding training after reaching 54 % accuracy")
			self.model.stop_training = True
			
callbacks = myCallback()

for fname in list_of_files:
	
	# Generate complete set_of_club_names:

	with open(fname, 'r') as csvfile:
		gamereader = csv.reader(csvfile)
		header = next(gamereader)
		for row in gamereader:
			number_of_games += 1
			if row[3] not in set_of_club_names:
				set_of_club_names.add(row[3])
			if row[4] not in set_of_club_names:
				set_of_club_names.add(row[4])
				
number_of_clubs = len(set_of_club_names)
list_of_club_names = list(set_of_club_names) # Needs to be accessible via index, hence conversion of set into list
				
# set_of_club_names is now a set of all unique club names in dataset
# number_of_games is now number of games in dataset

# Generate training data arrays:

x_data = np.zeros(shape=(number_of_games, 2, number_of_clubs), dtype=int)
y_data = np.zeros(shape=(number_of_games, 3), dtype=int)
row_number = 0

for fname in list_of_files:

	with open(fname, 'r') as csvfile:
		gamereader = csv.reader(csvfile)
		header = next(gamereader)
		for row in gamereader:
			home_team = list_of_club_names.index(row[3])
			away_team = list_of_club_names.index(row[4])
			x_data[row_number][0][home_team] = 1
			x_data[row_number][1][away_team] = 1
			goals_home = row[5]
			goals_away = row[6]
			if goals_home > goals_away:
				y_data[row_number][0] = 1
			if goals_home < goals_away:
				y_data[row_number][2] = 1
			if goals_home == goals_away:
				y_data[row_number][1] = 1
			row_number += 1

# x_data is now a 3D array: One dimension is the number of games analysed, the other two are two rows representing the home and away teams. Each row
# has as many elements as clubs in the set, and the home and away teams have, in the respective rows, values of 1

# y_data is now an array with the results of each game, where [1 0 0 ] means home team wins, [0 0 1] means away team wins, and [0 1 0] is a draw


model = tf.keras.Sequential([		# The model itself
	tf.keras.layers.Flatten(input_shape=(2, number_of_clubs)),
	tf.keras.layers.Dense(128),
	tf.keras.layers.Dense(512),
	tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

model.fit(x_data, y_data, epochs=250, callbacks=[callbacks])

# Print out a list of teams in the dataset and the keys to access them:
print()
k = 0
for club in list_of_club_names:
	print(f"#{k}: {club}")
	k += 1
print()

# Get prediction for this pairing
while True:
	# Ask user for a pairing to predict
	home_team_predict = int(input("Index number of home team for game to predict: "))
	away_team_predict = int(input("Index number of away team for game to predict: "))
	sample_to_predict = np.zeros(shape=(1, 2, number_of_clubs), dtype=int)
	
	sample_to_predict[0][0][home_team_predict] = 1
	sample_to_predict[0][1][away_team_predict] = 1
	prediction = model(sample_to_predict, training=False)
	prob_home = float(prediction[0][0])
	prob_draw = float(prediction[0][1])
	prob_away = float(prediction[0][2])
	print("\nPrediction: ")
	print(f"Victory for home team: {round((prob_home * 100), 2)} %")
	print(f"Draw: {round((prob_draw * 100), 2)} %")
	print(f"Victory for away team: {round((prob_away * 100), 2)} %")
	print(f"Based on {number_of_games} games.")
	#print("Prediction: ", prediction)
	#print("Sample: ", sample_to_predict)
	again = input("\nAnother game to predict (y/n)? ")
	if again == "n" or again =="N":
		break
		
savethis = input("Save this model (y/n)? ")
if savethis == "y" or savethis == "Y":
	# The following lines generate the HTML code for the select menus in the website
	generated_code = ""
	sorted_list_of_club_names = sorted(list_of_club_names)
	for club in sorted_list_of_club_names:
		generated_code += f"<option value='{list_of_club_names.index(club)}'>{club}</option>\n"
		
	with open('clubs.txt', 'w') as file:
		file.write(generated_code)
	tfjs.converters.save_keras_model(model, "/Users/Internet/Desktop/Tensorflow Experiments/Bundesliga/")
