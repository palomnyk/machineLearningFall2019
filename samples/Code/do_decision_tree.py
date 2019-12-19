# Author: Xiuxia Du
# Apring 2019

# Source 1: http://www.cse.msu.edu/~ptan/dmbook/tutorials/tutorial6/tutorial6.html
# Source 2: https://scikit-learn.org/stable/modules/tree.html

import os
import numpy as np
import pandas as pd
from sklearn import tree

import graphviz # for visualizing decision tree graph

# -----------------------------------------------------
# plotting setting
# -----------------------------------------------------
# for using latex
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rc('font', family='serif')

fig_width = 8
fig_height = 6

line_width = 2
marker_size = 26

axis_label_font_size = 9
legend_font_size = 9

# my customization of plotting
plot_params = {'figure.figsize': (fig_width, fig_height)}
plt.rcParams.update(plot_params)

# -----------------------------------------------------
# main
# -----------------------------------------------------
def main():
    index_example = 2
    # index_example = 1: use vertebrate data
    # index_example = 2: use iris data

    if index_example == 1:
        # import data
        data = pd.read_csv('../data/data_from_intro_to_data_mining/vertebrate.csv', header='infer')

        # change data table for doing binary classification
        data['Class'] = data['Class'].replace(['fishes', 'birds', 'amphibians', 'reptiles'], 'non-mammals')

        # apply Pandas cross-tabulation to examine the relationship between the Warm-blooded and
        # Gives Birth attributes with respect to the class.
        pd.crosstab([data['Warm-blooded'], data['Gives Birth']], data['Class'])

        # The results above show that it is possible to distinguish mammals from non-mammals using
        # these two attributes alone since each combination of their attribute values would yield
        # only instances that belong to the same class. For example, mammals can be identified as
        # warm-blooded vertebrates that give birth to their young. Such a relationship can also be
        # derived using a decision tree classifier, as shown by the example given in the next subsection.

        Y = data['Class']
        X = data.drop(['Name', 'Class'], axis=1)
        obj_DT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        obj_DT = obj_DT.fit(X, Y)

        testData = [['gila monster', 0, 0, 0, 0, 1, 1, 'non-mammals'],
                    ['platypus', 1, 0, 0, 0, 1, 1, 'mammals'],
                    ['owl', 1, 0, 0, 1, 1, 0, 'non-mammals'],
                    ['dolphin', 1, 1, 1, 0, 0, 0, 'mammals']]
        testData = pd.DataFrame(testData, columns=data.columns)

        testY = testData['Class']
        testX = testData.drop(['Name', 'Class'], axis=1)
        predY = obj_DT.predict(testX)
        predictions = pd.concat([testData['Name'], pd.Series(predY, name='Predicted Class')], axis=1)

        from sklearn.metrics import accuracy_score
        print('Accuracy on test data is %.2f' % (accuracy_score(testY, predY)))

    elif index_example == 2:
        # use iris data
        from sklearn.datasets import load_iris

        iris = load_iris()

        impurity_measure = 'gini'   # 'entropy' or 'gini'
        maximum_DT_depth = 2

        obj_DT = tree.DecisionTreeClassifier(criterion=impurity_measure, max_depth=maximum_DT_depth)
        obj_DT = obj_DT.fit(iris.data, iris.target)

        dot_data = tree.export_graphviz(obj_DT, out_file=None,
                                        feature_names=iris.feature_names,
                                        class_names=iris.target_names,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)

        out_path = "../results/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        out_file_name = "DT_graph_" + impurity_measure + "_" + str(maximum_DT_depth)
        out_file_full_name = os.path.join(out_path, out_file_name)

        graph.render(out_file_full_name)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(iris.data[0:50, 3], iris.data[0:50, 2], color='blue', s=marker_size, label='setosa')
        ax.scatter(iris.data[50:100, 3], iris.data[50:100, 2], color='red', s=marker_size, label='versicolor')
        ax.scatter(iris.data[100:150, 3], iris.data[100:150, 2], color='green', s=marker_size, label='virginica')
        ax.set_xlabel('petal width')
        ax.set_ylabel('petal length')
        ax.set_title('iris data')
        ax.legend()
        fig.show()

    else:
        print("\n Unknown data to use!\n")

        exit()

    xx = 1

if __name__ == '__main__':
    main()