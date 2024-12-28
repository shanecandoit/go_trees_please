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

	// we assume that the last column is the target column
	// and the rest are the features
	// we will convert the data to float32
	// and store the features in X and the target in y

	// process the data
	// TODO how to handle missing values?
	// for each row
	for i, record := range records {
		rowXs := []float32{}

		if i == 0 && !keepFirstRow {
			continue
		}

		// fmt.Println("record:", record)
		// for each column
		for colIndex, value := range record {
			// fmt.Printf("%s\t", value)

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
	dt.X, dt.y = shuffle(dt.X, dt.y)

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
	folds := SplitData(dt, k)

	for i := 0; i < k; i++ {

		fmt.Println("Fold:", i)
		dt_fold := folds[i]
		fmt.Println("Train:", dt_fold.X, dt_fold.y)

		// train the model
		// test the model
	}
}

func shuffle(X [][]float32, y []float32) ([][]float32, []float32) {
	// Seed the random number generator for reproducibility
	// rand.Seed(0)  // time.Now().UnixNano())

	// Create a permutation of indices
	n_rows := len(y)
	permutation := rand.Perm(n_rows)

	// n_cols := len(X[0])

	// Shuffle X and y according to the permutation
	shuffledX := make([][]float32, n_rows)
	shuffledY := make([]float32, n_rows)
	for i, j := range permutation {
		srcX := X[j]
		srcY := y[j]
		// shuffledX = append(shuffledX, srcX)
		// shuffledY = append(shuffledY, srcY)
		shuffledX[i] = srcX
		shuffledY[i] = srcY
	}

	// Replace original X and y with shuffled versions
	// copy(X, shuffledX)
	// copy(y, shuffledY)
	return shuffledX, shuffledY
}

// SplitData splits the data into k-folds for cross-validation
func SplitData(data DataTable, k int) []DataTable {
	n_rows := len(data.y)
	// n_cols := len(data.X[0])
	folds := make([]DataTable, k)
	// n_folds_rows := n_rows / k

	// Initialize empty folds
	for i := range folds {
		folds[i] = DataTable{
			X: [][]float32{},
			y: []float32{},
		}
	}

	// Create shuffled indices
	indices := rand.Perm(n_rows)

	// Distribute data points into folds
	for i, idx := range indices {
		foldIndex := i % k

		srcX := data.X[idx]
		srcY := data.y[idx]

		// Append data point to appropriate fold
		folds[foldIndex].X = append(folds[foldIndex].X, srcX)
		folds[foldIndex].y = append(folds[foldIndex].y, srcY)

		// folds[foldIndex].X[i] = make([]float32, len(srcX))
		// folds[foldIndex].X[i] = srcX
		// folds[foldIndex].y[i] = srcY
	}

	// print first row of each fold
	for i := 0; i < k; i++ {
		fmt.Print("Fold:", i)
		fmt.Print(", X:", folds[i].X[0])
		fmt.Println(", y:", folds[i].y[0])
	}

	return folds
}
