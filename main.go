package main

import (
	"fmt"
	"log"
	"math"
)

func main() {
	fmt.Println("Hello, World!")

	dataSet := DataSetFromCSV("examples/iris.csv")
	trainData := dataSet.train
	testData := dataSet.test

	fmt.Println("trainData", trainData)
	fmt.Println("testData", testData)

	tree := TreeNew()
	tree.Fit(trainData.X, trainData.y)
	tree.Print()

	score := tree.Score(testData.X, testData.y)
	fmt.Println("Score:", score)
}

// Tree is a struct that represents a decision tree, or a collection of decision trees
type Tree struct {
	n_estimators int
	max_depth    int

	Root *TreeNode
}

// DataTable is a struct that represents a dataset for training a decision tree model
type DataTable struct {
	X [][]float32
	y []float32
}

// DataSet is a struct that represents a dataset for training and testing a decision tree model
type DataSet struct {
	train DataTable
	test  DataTable
}

// DataSetFromCSV is a method that loads a dataset from a CSV file
func DataSetFromCSV(path string) *DataSet {
	// load from csv
	dt, err := LoadCSV(path)
	if err != nil {
		log.Fatal(err)
	}

	// split the data into train and test
	train, test := SplitDataTrainTest(dt, 0.8)
	ds := DataSet{
		train: train,
		test:  test,
	}
	return &ds
}

// New is a method that creates a new decision tree model
// With default values for n_estimators and max_depth
func TreeNew() *Tree {
	tree := Tree{
		n_estimators: 10,
		max_depth:    5,
	}
	return &tree
}

// New is a method that creates a new decision tree model
// With default values for n_estimators and max_depth
// func TreeFromCSV(path string) *Tree {
// 	tree := TreeNew()

// 	// load from csv
// 	dt, err := LoadCSV(path)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	// fit the tree
// 	tree.Fit(dt.X, dt.y)
// 	return tree
// }

func (t *Tree) Print() {
	fmt.Println("Tree:", t)
	fmt.Println("n_estimators:", t.n_estimators)
	fmt.Println("max_depth:", t.max_depth)

}

// Predict is a method that makes predictions using the decision tree model
func (t *Tree) Predict(X [][]float32) []float32 {
	predictions := make([]float32, len(X))
	for i, sample := range X {
		predictions[i] = t.predictSample(sample, t.Root)
	}
	return predictions
}

// Helper function to predict a single sample
func (t *Tree) predictSample(sample []float32, node *TreeNode) float32 {
	if node.IsLeaf {
		return node.Prediction
	}
	if sample[node.FeatureIndex] <= node.Threshold {
		return t.predictSample(sample, node.Left)
	}
	return t.predictSample(sample, node.Right)
}

// Score is a method that evaluates the decision tree model
func (t *Tree) Score(X [][]float32, y []float32) float32 {

	// make predictions
	predictions := t.Predict(X)

	// calculate the accuracy
	correct := 0
	for i, prediction := range predictions {
		if prediction == y[i] {
			correct++
		}
	}

	accuracy := float32(correct) / float32(len(y))
	return accuracy
}

// Fit is a method that trains the decision tree model
func (t *Tree) Fit(X [][]float32, y []float32) {
	// Train the decision tree model

	// make a new DataTable
	dt := DataTable{
		X: [][]float32{},
		y: []float32{},
	}
	// Copy X
	for i, row := range X {
		// dt.X[i] = make([]float32, len(row))
		// dt.X = append(dt.X, make([]float32, len(row)))
		// copy(dt.X[i], row)
		dt.X = append(dt.X, row)
		dt.y = append(dt.y, y[i])
	}
	// Copy y
	// dt.y = make([]float32, len(y))
	// copy(dt.y, y)
	// are the ys continuous or discrete?
	// if continuous, we are doing regression
	// if discrete, we are doing classification

	// shuffle the order
	dt.X, dt.y = Shuffle(dt.X, dt.y)

	if len(dt.X) != len(dt.y) {
		log.Fatal("X and y must have the same number of rows")
	}

	isDiscrete := true
	classCounts := make(map[float32]int)
	for i := 0; i < len(dt.y); i++ {

		// count the number of times each class appears
		classCounts[dt.y[i]]++

		// check if the y values are discrete
		if dt.y[i] != float32(int(dt.y[i])) {
			isDiscrete = false
			// break
		}
	}
	fmt.Println("isDiscrete", isDiscrete)
	fmt.Println("classCounts", classCounts)
	// TODO if classes are un-balanced, we may need to balance them?

	// split the data into k folds
	k := 5
	folds := SplitDataKParts(dt, k)
	// folds 1,2,3 are training
	// fold 4 is validation
	// fold 5 is test
	for i := 0; i < k; i++ {
		fmt.Println("Fold:", i, "len:", len(folds[i].X), len(folds[i].y))
		dt_fold := folds[i]
		// fmt.Println("Train:", dt_fold.X, dt_fold.y)

		// check for proper data
		if dt_fold.X == nil || dt_fold.y == nil {
			log.Fatal("Fold X and y must not be nil")
		}
		if len(dt_fold.X) != len(dt_fold.y) {
			log.Fatal("Fold X and y must have the same number of rows")
		}
	}

	// for i := 0; i < k; i++ {

	// 	fmt.Println("Fold:", i)
	// 	dt_fold := folds[i]
	// 	// fmt.Println("Train:", dt_fold.X, dt_fold.y)

	// 	// train the model
	// 	// test the model
	// }

	id3Tree := FitID3(dt, t.max_depth, isDiscrete)
	fmt.Println("id3Tree", id3Tree)

	t.Root = id3Tree

	// validate
	// set score
}

// FitID3 is a method that trains the decision tree model using the ID3 algorithm
func FitID3(dt DataTable, max_depth int, isDiscrete bool) *TreeNode {
	// Train the decision tree model using the ID3 algorithm
	root := Tree_ID3(dt, max_depth, isDiscrete)
	return root
}

// TreeNode is a struct that represents a node in a decision tree
type TreeNode struct {
	// Split criteria
	FeatureIndex int
	Threshold    float32

	// Tree structure
	Left   *TreeNode
	Right  *TreeNode
	Parent *TreeNode

	// Leaf information
	Prediction float32
	IsLeaf     bool

	// Metadata
	Depth       int
	SampleCount int
	Impurity    float32
}

// NewTreeNode is a method that creates a new tree node
func NewTreeNode() *TreeNode {
	return &TreeNode{
		FeatureIndex: -1,
		Threshold:    0.0,
		Left:         nil,
		Right:        nil,
		Parent:       nil,
		Prediction:   0.0,
		IsLeaf:       false,
		Depth:        0,
		SampleCount:  0,
		Impurity:     0.0,
	}
}

// Tree_ID3 is a method that trains a decision tree using the ID3 algorithm
func Tree_ID3(dt DataTable, max_depth int, isDiscrete bool) *TreeNode {
	// Train the decision tree model using the ID3 algorithm
	root := NewTreeNode()
	root.SampleCount = len(dt.y)
	root.Depth = 0
	root.Impurity = Gini(dt.y) // or Entropy(dt.y)

	// Check if we need to stop splitting
	if root.Depth >= max_depth || root.Impurity == 0.0 {
		root.IsLeaf = true
		root.Prediction = MajorityVote(dt.y)
		return root
	}

	// Find the best split
	bestSplit := FindBestSplit(dt, isDiscrete)
	if bestSplit == nil {
		root.IsLeaf = true
		root.Prediction = MajorityVote(dt.y)
		return root
	}

	// Split the data
	leftData, rightData := SplitData(dt, bestSplit.FeatureIndex, bestSplit.Threshold)

	// Recursively build the tree
	root.FeatureIndex = bestSplit.FeatureIndex
	root.Threshold = bestSplit.Threshold
	root.Left = Tree_ID3(leftData, max_depth, isDiscrete)
	root.Right = Tree_ID3(rightData, max_depth, isDiscrete)

	return root
}

func Gini(y []float32) float32 {
	// Calculate the Gini impurity of a dataset
	// Gini: 1 - Σ(pi²)
	// Gini range: [0, 0.5]
	classCounts := make(map[float32]int)
	for _, class := range y {
		classCounts[class]++
	}

	impurity := float32(1.0)
	for _, count := range classCounts {
		prob := float32(count) / float32(len(y))
		impurity -= prob * prob
	}

	fmt.Println("Gini: of ", len(y), "rows", impurity)

	return impurity
}

func Entropy(y []float32) float32 {
	// Entropy: -Σ(pi * log2(pi))
	// Entropy range: [0, 1]
	classCounts := make(map[float32]int)
	for _, class := range y {
		classCounts[class]++
	}

	entropy := float32(0.0)
	for _, count := range classCounts {
		prob := float32(count) / float32(len(y))
		if prob > 0 {
			entropy -= prob * float32(math.Log2(float64(prob)))
		}
	}

	return entropy
}

func MajorityVote(y []float32) float32 {
	// Find the majority class in a dataset
	classCounts := make(map[float32]int)
	for _, class := range y {
		classCounts[class]++
	}

	majorityClass := float32(0)
	maxCount := 0
	for class, count := range classCounts {
		if count > maxCount {
			majorityClass = class
			maxCount = count
		}
	}

	return majorityClass
}

func FindBestSplit(dt DataTable, isDiscrete bool) *Split {
	fmt.Println("FindBestSplit", len(dt.X), "discrete", isDiscrete)
	if len(dt.X) != len(dt.y) {
		log.Fatal("X and y must have the same number of rows")
	}
	if len(dt.X) == 0 || len(dt.y) == 0 {
		log.Fatal("X and y must not be empty")
	}
	// Find the best split for a dataset
	bestSplit := &Split{
		FeatureIndex: -1,
		Threshold:    0.0,
		Impurity:     1.0,
	}

	// Iterate over each feature
	for i := 0; i < len(dt.X[0]); i++ {
		// Iterate over each data point
		for j := 0; j < len(dt.X); j++ {
			threshold := dt.X[j][i]
			leftData, rightData := SplitData(dt, i, threshold)

			// Calculate the impurity of the split
			impurity := Impurity(leftData.y, rightData.y)
			fmt.Println("Feature:", i, "Threshold:", threshold, "Impurity:", impurity)

			// Update the best split if necessary
			if impurity < bestSplit.Impurity {
				bestSplit.FeatureIndex = i
				bestSplit.Threshold = threshold
				bestSplit.Impurity = impurity
			}
		}
	}

	return bestSplit
}

// Split is a struct that represents a split in a decision tree
type Split struct {
	FeatureIndex int
	Threshold    float32
	Impurity     float32
}

// Impurity is a method that calculates the impurity of a split
func Impurity(left []float32, right []float32) float32 {
	// Calculate the impurity of a split
	total := float32(len(left) + len(right))
	impurity := (float32(len(left)) / total) * Gini(left)
	impurity += (float32(len(right)) / total) * Gini(right)
	return impurity
}

// SplitData is a method that splits a dataset into two parts
// leftData, rightData := SplitData(dt, bestSplit.FeatureIndex, bestSplit.Threshold)
func SplitData(dt DataTable, featureIndex int, threshold float32) (DataTable, DataTable) {
	// Split a dataset into two parts
	leftData := DataTable{
		X: [][]float32{},
		y: []float32{},
	}
	rightData := DataTable{
		X: [][]float32{},
		y: []float32{},
	}

	for i := 0; i < len(dt.X); i++ {
		if dt.X[i][featureIndex] < threshold {
			leftData.X = append(leftData.X, dt.X[i])
			leftData.y = append(leftData.y, dt.y[i])
		} else {
			rightData.X = append(rightData.X, dt.X[i])
			rightData.y = append(rightData.y, dt.y[i])
		}
	}

	return leftData, rightData
}
