package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
)

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

func Shuffle(X [][]float32, y []float32, seed *int) ([][]float32, []float32) {
	// Seed the random number generator for reproducibility
	// rand.Seed(0)  // time.Now().UnixNano())

	// if seed is nil, generate a random seed
	if seed == nil {
		num := rand.Int()
		seed = &num
	}
	rand.New(rand.NewSource(int64(*seed)))

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

// SplitDataKParts splits the data into k-folds for cross-validation
func SplitDataKParts(data *DataTable, k int) []DataTable {
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

// // split the data into train and test
// train, test := SplitDataTrainTest(dt, 0.8)
func SplitDataTrainTest(data *DataTable, trainRatio float64) (DataTable, DataTable) {
	fmt.Println("SplitDataTrainTest")
	n_rows := len(data.y)
	fmt.Println("n_rows:", n_rows)
	// n_cols := len(data.X[0])
	trainRows := int(float64(n_rows) * trainRatio)
	fmt.Println("trainRows:", trainRows)
	// testRows := n_rows - trainRows

	// Create empty DataTables
	trainData := DataTable{
		X: [][]float32{},
		y: []float32{},
	}
	testData := DataTable{
		X: [][]float32{},
		y: []float32{},
	}

	// Copy data into train and test sets
	trainData.X = data.X[:trainRows]
	trainData.y = data.y[:trainRows]
	testData.X = data.X[trainRows:]
	testData.y = data.y[trainRows:]

	return trainData, testData
}
