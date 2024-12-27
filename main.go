package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
)

func main() {
	fmt.Println("Hello, World!")

	tree := TreeFromCSV("examples/iris.csv")
	tree.Print()
}

func LoadCSV(path string) (*DataTable, error) {

	// open the csv file
	openFile, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	defer openFile.Close()
	// read the csv file
	reader := csv.NewReader(openFile)
	// data
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		return nil, err
	}

	// columns and rows
	columns := len(records[0])
	fmt.Println("columns:", columns)
	rows := len(records)
	fmt.Println("rows:", rows)

	// is the first row a header?
	// if so, we will skip it
	firstRowStringCount := 0
	firstRowNumberCount := 0
	for _, value := range records[0] {
		_, err := strconv.ParseFloat(value, 64)
		if err == nil {
			firstRowNumberCount++
		} else {
			firstRowStringCount++
		}
	}
	fmt.Println("firstRowStringCount:", firstRowStringCount)
	fmt.Println("firstRowNumberCount:", firstRowNumberCount)

	keepFirstRow := false
	if firstRowStringCount == 0 {
		keepFirstRow = true
	}
	fmt.Println("keepFirstRow:", keepFirstRow)

	targetColumn := columns - 1

	// create a new DataTable
	dt := DataTable{
		X: [][]float32{},
		y: []float32{},
	}

	rowXs := []float32{}

	// we assume that the last column is the target column
	// and the rest are the features
	// we will convert the data to float32
	// and store the features in X and the target in y

	// process the data
	// TODO how to handle missing values?
	// for each row
	for i, record := range records {

		if i == 0 && !keepFirstRow {
			continue
		}

		// fmt.Println("record:", record)
		// for each column
		for colIndex, value := range record {
			fmt.Printf("%s\t", value)

			if colIndex == targetColumn {
				y, err := strconv.ParseFloat(value, 32)
				y32 := float32(y)
				if err != nil {
					log.Fatal(err)
				}
				// add target to y
				dt.y = append(dt.y, y32)
			} else {
				x, err := strconv.ParseFloat(value, 32)
				x32 := float32(x)
				if err != nil {
					log.Fatal(err)
				}
				rowXs = append(rowXs, x32)
			}

		}
		// add features to X
		dt.X = append(dt.X, rowXs)
	}

	return &dt, nil

}

// Tree is a struct that represents a decision tree, or a collection of decision trees
type Tree struct {
	n_estimators int
	max_depth    int
}

// DataTable is a struct that represents a dataset for training a decision tree model
type DataTable struct {
	X [][]float32
	y []float32
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
func TreeFromCSV(path string) *Tree {
	tree := TreeNew()

	// load from csv
	dt, err := LoadCSV(path)
	if err != nil {
		log.Fatal(err)
	}

	// fit the tree
	tree.Fit(dt.X, dt.y)
	return tree
}

func (t *Tree) Print() {
	fmt.Println("Tree:", t)
	fmt.Println("n_estimators:", t.n_estimators)
	fmt.Println("max_depth:", t.max_depth)

}

// Predict is a method that makes predictions using the decision tree model
func (t *Tree) Predict(X [][]float32) []float32 {
	// TODO: Implement the predict method
	return []float32{}
}

// Score is a method that evaluates the decision tree model
func (t *Tree) Score(X [][]float32, y []float32) float32 {
	// TODO: Implement the score method
	return 0.0
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
		dt.X = append(dt.X, make([]float32, len(row)))
		copy(dt.X[i], row)
	}
	// Copy y
	dt.y = make([]float32, len(y))
	copy(dt.y, y)
	// are the ys continuous or discrete?
	// if continuous, we are doing regression
	// if discrete, we are doing classification

	// shuffle the order
	shuffle(dt.X, dt.y)

	isDiscrete := true
	classCounts := make(map[float32]int)
	for i := 0; i < len(dt.y); i++ {

		// count the number of times each class appears
		classCounts[dt.y[i]]++

		// check if the y values are discrete
		if y[i] != float32(int(dt.y[i])) {
			isDiscrete = false
			// break
		}
	}

	fmt.Println("isDiscrete", isDiscrete)

	// split the data into k folds
	k := 5
	folds := SplitData(dt, k)

	for i := 0; i < k; i++ {

		fmt.Println("Fold:", i)
		dt_fold := folds[i]
		fmt.Println("Train:", dt_fold.X, dt_fold.y)

		// train the model
		// test the model
	}
}

func shuffle(X [][]float32, y []float32) {
	// Seed the random number generator for reproducibility
	// rand.Seed(0)  // time.Now().UnixNano())

	// Create a permutation of indices
	n := len(y)
	permutation := rand.Perm(n)

	// Shuffle X and y according to the permutation
	shuffledX := make([][]float32, n)
	shuffledY := make([]float32, n)
	for i, j := range permutation {
		shuffledX[i] = X[j]
		shuffledY[i] = y[j]
	}

	// Replace original X and y with shuffled versions
	copy(X, shuffledX)
	copy(y, shuffledY)
}

// SplitData splits the data into k-folds for cross-validation
func SplitData(data DataTable, k int) []DataTable {
	// rand.Seed(time.Now().UnixNano())

	// Create a slice of DataTables to hold the folds
	// type DataTable struct {X [][]float32; y []float32 }

	totalRows := len(data.y)

	// Shuffle the data indices
	// indices := rand.Perm(len(data.y))
	indices := make([]int, totalRows)
	fmt.Println("indices:", indices)

	for i := 0; i < totalRows; i++ {
		fmt.Println("i:", i)
		randIndex := rand.Intn(k)
		indices[i] = randIndex
	}

	// Create a slice of DataTables to hold the folds
	folds := make([]DataTable, k)

	// Split the data into k folds
	// copy DataTable.X into filds[indices[i]].X
	// copy DataTable.y into filds[indices[i]].y

	return folds
}
