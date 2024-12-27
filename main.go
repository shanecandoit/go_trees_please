package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

func main() {
	fmt.Println("Hello, World!")

	tree := TreeFromCSV("examples/iris.csv")
	tree.Print()
}

func LoadCSV(path string) (*DataTable, error) {
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
		X: [][]float64{},
		y: []float64{},
	}

	rowXs := []float64{}

	// we assume that the last column is the target column
	// and the rest are the features
	// we will convert the data to float64
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
				y, err := strconv.ParseFloat(value, 64)
				if err != nil {
					log.Fatal(err)
				}
				// add target to y
				dt.y = append(dt.y, y)
			} else {
				x, err := strconv.ParseFloat(value, 64)
				if err != nil {
					log.Fatal(err)
				}
				rowXs = append(rowXs, x)
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
	X [][]float64
	y []float64
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
	dt, err := LoadCSV(path)
	if err != nil {
		log.Fatal(err)
	}
	tree.Fit(dt.X, dt.y)
	return tree
}

func (t *Tree) Print() {
	fmt.Println("Tree:", t)
	fmt.Println("n_estimators:", t.n_estimators)
	fmt.Println("max_depth:", t.max_depth)

}

// Fit is a method that trains the decision tree model
func (t *Tree) Fit(X [][]float64, y []float64) {
	// Train the decision tree model

	// make a new DataTable
	dt := DataTable{
		X: [][]float64{},
		y: []float64{},
	}
	// are the ys continuous or discrete?
	// if continuous, we are doing regression
	// if discrete, we are doing classification

	isDiscrete := true
	classCounts := make(map[float64]int)
	for i := 0; i < len(dt.y); i++ {

		// count the number of times each class appears
		classCounts[dt.y[i]]++

		// check if the y values are discrete
		if y[i] != float64(int(dt.y[i])) {
			isDiscrete = false
			// break
		}
	}

	fmt.Println("isDiscrete", isDiscrete)
}
