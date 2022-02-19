# SmartificialSmintelligence
Toying around a little with machine learning

## game_predicter.py ##

This is a tool to start dabbling around in machine learning with TensorFlow/Keras. It uses data on past results of sports leagues (I used the German soccer Bundesliga for the seasons 2017/18 through 2021/22 for this purpose, representing a total of 1413 games) to arrive at predictions for the outcome of an upcoming game.

The programme loads the data from files (by default in the same folder as the .py file, and the file names are hardcoded, but of course file paths can easily be changed by the user). This data (team pairings and results, i.e. home team wins/draw/away teams wins) is fed through a three-layer neural network. The user is then prompted to enter a hypothetical pairing of two teams in the dataset (the entries are made by number rather than name, to facilitate typing and reduce the risk of mistyping); the model will return a prediction of probabilities for the three possible outcomes (home team wins/draw/away teams wins).

With the Bundesliga data I used, I get a training data accuracy of somewhere in the 50-55 % range, which is not great but significantly better than random guesses. A callback is in place to end training when the model reaches 55 % accuracy. There is no validation data, but you can, of course, use the model to predict upcoming games and validate on that basis.

Version 1.5, February 2022
