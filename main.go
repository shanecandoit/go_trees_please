package main

import (
	"fmt"
	"log"
	"math/rand"
)

func main() {
	fmt.Println("Hello, World!")

	// dataSet := DataSetFromCSV("examples/iris.csv")
	// trainData := dataSet.train

	// fmt.Println("trainData", trainData)
	// fmt.Println("testData", testData)

	// tree := TreeNew()
	// tree.Fit(trainData.X, trainData.y)
	var seed int = 1
	tree, dataSet := TreeFromCSV("examples/iris.csv", &seed)
	testData := dataSet.test
	tree.Print()

	tree.SaveToDotFile("my_tree.dot")

	score := tree.Score(testData.X, testData.y)
	fmt.Println("Score:", score)
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

func TreeFromCSV(path string, seed *int) (*Tree, *DataSet) {

	// if seed is nil, generate a random seed
	if seed == nil {
		num := rand.Int()
		seed = &num
	}
	seedInt64 := int64(*seed)
	rand.New(rand.NewSource(seedInt64))
	// TODO: make seed work right

	tree := TreeNew()
	tree.RandomState = seed

	dataSet := DataSetFromCSV("examples/iris.csv")
	trainData := dataSet.train
	testData := dataSet.test

	tree.DataTrainSet = &trainData

	fmt.Println("trainData", trainData)
	fmt.Println("testData", testData)

	// tree.Fit(trainData.X, trainData.y)
	tree.Fit()

	return tree, dataSet
}

// SplitData is a method that splits a dataset into two parts
// leftData, rightData := SplitData(dt, bestSplit.FeatureIndex, bestSplit.Threshold)
func SplitData(dt *DataTable, featureIndex int, threshold float32) (DataTable, DataTable) {
	// Split a dataset into two parts
	// left child: X < threshold, right child: X >= threshold
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
