package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	myApp := app.New()
	myWindow := myApp.NewWindow("NeurOG")

	// Criar rótulos
	labelTrain := widget.NewLabel("Nome do arquivo de treinamento")
	labelTest := widget.NewLabel("Nome do arquivo de teste")
	labelLR := widget.NewLabel("Learning Rate")
	labelEpochs := widget.NewLabel("Número de Épocas")
	labelHiddenNeurons := widget.NewLabel("Neurônios na Camada Oculta")

	// Criar entradas
	entryTrain := widget.NewEntry()
	entryTrain.SetPlaceHolder("teste.csv")

	entryTest := widget.NewEntry()
	entryTest.SetPlaceHolder("treinamento.csv")

	entryLR := widget.NewEntry()
	entryLR.SetPlaceHolder("0.3")

	entryEpochs := widget.NewEntry()
	entryEpochs.SetPlaceHolder("5000")

	entryHiddenNeurons := widget.NewEntry()
	entryHiddenNeurons.SetPlaceHolder("25")

	// Criar layout para organizar os elementos
	form := container.NewVBox(
		container.New(layout.NewVBoxLayout(),
			labelTrain,
			entryTrain,
			labelTest,
			entryTest,
			labelLR,
			entryLR,
			labelEpochs,
			entryEpochs,
			labelHiddenNeurons,
			entryHiddenNeurons,
			widget.NewButton("Treinar e Testar", func() {
				// Obtenha os valores dos campos de entrada
				trainFileName := entryTrain.Text
				testFileName := entryTest.Text

				lr, err := strconv.ParseFloat(entryLR.Text, 64)
				if err != nil {
					log.Fatal(err)
					return
				}

				epochs, err := strconv.Atoi(entryEpochs.Text)
				if err != nil {
					log.Fatal(err)
					return
				}

				hiddenNeurons, err := strconv.Atoi(entryHiddenNeurons.Text)
				if err != nil {
					log.Fatal(err)
					return
				}

				// Chame a função trainAndTest
				trainAndTest(trainFileName, testFileName, hiddenNeurons, epochs, lr)
			}),
		),
	)

	myWindow.SetContent(container.NewVBox(
		widget.NewLabel("NeurOG"), // Título
		form,
	))
	myWindow.Resize(fyne.NewSize(600, 300))
	myWindow.SetMaster()

	myWindow.ShowAndRun()
}

func trainAndTest(trainFileName, testFileName string, hiddenNeurons, numEpochs int, learningRate float64) {
	// lê os arquivos de treinamento
	inputs, labels := makeInputsAndLabels(trainFileName)
	labels = formatLabel(labels)

	config := neuralNetConfig{
		inputNeurons:  6,
		outputNeurons: 5,
		hiddenNeurons: hiddenNeurons,
		numEpochs:     numEpochs,
		learningRate:  learningRate,
	}

	// Cria e  treina a rede neural
	network := createNeuralNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Formaliza a matriz de teste
	testInputs, testLabels := makeInputsAndLabels(testFileName)
	testLabels = formatLabel(testLabels)

	// Realiza as predições usando o modelo treinado (network)
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	binarizedPredictions := binarizePredictions(predictions)
	fmt.Println("binarized Predictions\n", mat.Formatted(binarizedPredictions, mat.Squeeze()))
}

// Função para transformar as predições em uma matriz binária
func binarizePredictions(predictions *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	binarized := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		// Encontra o índice do maior elemento na linha
		_, maxIdx := findMaxIndex(predictions.RowView(i))
		for j := 0; j < cols; j++ {
			// Define 1 no índice do maior elemento, 0 nos demais
			if j == maxIdx {
				binarized.Set(i, j, 1)
			} else {
				binarized.Set(i, j, 0)
			}
		}
	}

	return binarized
}

// Função para encontrar o índice do valor máximo em um vetor
func findMaxIndex(v mat.Vector) (float64, int) {
	maxIdx := 0
	maxVal := v.AtVec(0)

	for i := 1; i < v.Len(); i++ {
		val := v.AtVec(i)
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxVal, maxIdx
}

// Define a arquitetura e os parâmetros de aprendizado da Rede Neural
type neuralNetConfig struct {
	inputNeurons  int     // Quantidade de neurônios de entrada
	outputNeurons int     // Quantidade de neurônios de saída
	hiddenNeurons int     // Quantidade de neurônios escondidos
	numEpochs     int     // Quantidade de "Epochs"
	learningRate  float64 // Taxa de aprendizado
}

// Contém as informações sobre o treinamento da rede neural
type neuralNet struct {
	config       neuralNetConfig // Configurações
	weightHidden *mat.Dense      // Matriz de pesos de entrada
	biasHidden   *mat.Dense      // Matriz de bias de entrada
	weightOut    *mat.Dense      // Matriz de pesos de saída
	biasOut      *mat.Dense      // Matriz de bias de saída
}

// Inicia uma nova rede neural
func createNeuralNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// Implementação da sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// implementação da sigmoid derivativa para o backpropagation
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// Formata o label para 1,0,0,0,0 / 0,1,0,0,0 / ...
func formatLabel(labels *mat.Dense) *mat.Dense {
	rows, cols := labels.Dims()
	labelsOutput := mat.NewDense(rows, cols*5, nil)

	for i := 0; i < rows; i++ {
		val := int(labels.At(i, 0)) - 1 // Ajuste para começar de 0
		for j := 0; j < cols*5; j++ {
			if j%5 == val {
				labelsOutput.Set(i, j, 1)
			} else {
				labelsOutput.Set(i, j, 0)
			}
		}
	}

	return labelsOutput
}

// Treina a rede neural
func (nn *neuralNet) train(inputs, labels *mat.Dense) error {

	rand.Seed(time.Now().UnixNano())

	// Inicializa os pesos e bias
	weightInputHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	biasInputHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	weightHiddenOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	biasHiddenOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	weightInputHiddenRaw := weightInputHidden.RawMatrix().Data
	biasInputHiddenRaw := biasInputHidden.RawMatrix().Data
	weightHiddenOutRaw := weightHiddenOut.RawMatrix().Data
	biasHiddenOutRaw := biasHiddenOut.RawMatrix().Data

	// Atribui valores aleatórios aos pesos e bias
	for _, param := range [][]float64{
		weightInputHiddenRaw,
		biasInputHiddenRaw,
		weightHiddenOutRaw,
		biasHiddenOutRaw,
	} {
		for i := range param {
			param[i] = rand.Float64()*2 - 1
		}
	}

	fmt.Println("weightInputHidden\n", mat.Formatted(weightInputHidden, mat.Squeeze()))
	fmt.Println("biasInputHidden\n", mat.Formatted(biasInputHidden, mat.Squeeze()))
	fmt.Println("weightHiddenOut\n", mat.Formatted(weightHiddenOut, mat.Squeeze()))
	fmt.Println("biasHiddenOut\n", mat.Formatted(biasHiddenOut, mat.Squeeze()))

	// Saída da rede neural
	output := new(mat.Dense)

	// Usa o backpropagation para ajustar os pesos e bias
	if err := nn.backPropagate(
		inputs,
		labels,
		weightInputHidden,
		biasInputHidden,
		weightHiddenOut,
		biasHiddenOut,
		output); err != nil {
		return err
	}

	// FIM DO TREINAMENTO: Implementa os elementos dentro da rede neural
	nn.weightHidden = weightInputHidden
	nn.biasHidden = biasInputHidden
	nn.weightOut = weightHiddenOut
	nn.biasOut = biasHiddenOut

	return nil
}

func (nn *neuralNet) backPropagate(inputs,
	labels,
	weightInputHidden,
	biasInputHidden,
	weightHiddenOut,
	biasHiddenOut,
	output *mat.Dense) error {

	// Loop através do número de epochs utilizando backpropagation
	for i := 0; i < nn.config.numEpochs; i++ {

		// FEEDFORWARD
		// Input -> hidden
		hiddenLayerInput := new(mat.Dense)
		// fmt.Println("inputs\n", mat.Formatted(inputs, mat.Squeeze()))
		// fmt.Println("weightInputHidden\n", mat.Formatted(weightInputHidden, mat.Squeeze()))
		hiddenLayerInput.Mul(inputs, weightInputHidden)
		// fmt.Println("hiddenLayerInput\n", mat.Formatted(hiddenLayerInput, mat.Squeeze()))
		addBiasInputHidden := func(_, col int, v float64) float64 {
			return v + biasInputHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBiasInputHidden, hiddenLayerInput)
		// Aplicação da sigmoid
		InputHiddenLayerActivation := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		InputHiddenLayerActivation.Apply(applySigmoid, hiddenLayerInput)
		// fmt.Println("InputHiddenLayerActivation\n", mat.Formatted(InputHiddenLayerActivation, mat.Squeeze()))

		// hidden -> output
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(InputHiddenLayerActivation, weightHiddenOut)
		addBiasHiddenOut := func(_, col int, v float64) float64 {
			return v + biasHiddenOut.At(0, col)
		}
		outputLayerInput.Apply(addBiasHiddenOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Backpropagation
		// fmt.Println(labels.Dims())
		// fmt.Println("labels:\n", mat.Formatted(labels, mat.Squeeze()))
		// fmt.Println(output.Dims())
		// fmt.Println("output:\n", mat.Formatted(output, mat.Squeeze()))
		networkError := new(mat.Dense)
		networkError.Sub(labels, output)
		// fmt.Println("Error:\n", mat.Formatted(networkError, mat.Squeeze()))

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, InputHiddenLayerActivation)
		//
		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, weightHiddenOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Ajusta os parâmetros
		weightOutAdj := new(mat.Dense)
		weightOutAdj.Mul(InputHiddenLayerActivation.T(), dOutput)
		weightOutAdj.Scale(nn.config.learningRate, weightOutAdj)
		weightHiddenOut.Add(weightHiddenOut, weightOutAdj)

		biasOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		biasOutAdj.Scale(nn.config.learningRate, biasOutAdj)
		biasHiddenOut.Add(biasHiddenOut, biasOutAdj)

		weightHiddenAdj := new(mat.Dense)
		weightHiddenAdj.Mul(inputs.T(), dHiddenLayer)
		weightHiddenAdj.Scale(nn.config.learningRate, weightHiddenAdj)
		weightInputHidden.Add(weightInputHidden, weightHiddenAdj)

		biasHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		biasHiddenAdj.Scale(nn.config.learningRate, biasHiddenAdj)
		biasInputHidden.Add(biasInputHidden, biasHiddenAdj)

	}
	fmt.Println(output.Dims())
	fmt.Println("output\n", mat.Formatted(output, mat.Squeeze()))

	return nil
}

// sumAlongAxis soma uma matriz ao longo de uma dimensão específica,
// preservando a outra dimensão.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// Implementação do feed forward para previsão
// predict faz uma previsão com base em uma rede
// neural treinada.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Verifica se o valor neuralNet representa um modelo treinado.
	if nn.weightHidden == nil || nn.weightOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.biasHidden == nil || nn.biasOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Defina a saída da rede neural.
	output := new(mat.Dense)

	// Completa o processo de feed forward.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.weightHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.biasHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.weightOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.biasOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// Lê o CSV
func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Abra o arquivo do conjunto de dados.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Crie um novo leitor de CSV lendo do arquivo aberto.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Leia todos os registros CSV.
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData e labelsData irão conter todos os
	// valores de ponto flutuante que eventualmente serão
	// usados para formar matrizes.
	inputsData := make([]float64, 6*len(rawCSVData))
	labelsData := make([]float64, 1*len(rawCSVData))

	// Irá rastrear o índice atual dos valores da matriz.
	var inputsIndex int
	var labelsIndex int

	// Move sequentially through the rows to a slice of floating-point values.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Iterate over the floating-point columns.
		for i, val := range record {

			// Convert the value to float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Convert the value to a floating-point.
			if i == 6 { // Assuming classe is the last column (index 6)
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add to inputsData if relevant.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 6, inputsData)
	labels := mat.NewDense(len(rawCSVData), 1, labelsData)
	return inputs, labels
}

// Informações
// ShowModelInfo exibe informações sobre a rede neural após o treinamento.
func ShowModelInfo(network *neuralNet) {
	colorReset := "\033[0m"
	colorYellow := "\033[33m"

	fmt.Println(string(colorYellow))
	fmt.Println("Informações da Rede Neural APÓS O TREINAMENTO:")
	fmt.Printf("Arquitetura da Rede: %d -> %d -> %d\n", network.config.inputNeurons, network.config.hiddenNeurons, network.config.outputNeurons)
	fmt.Printf("Número de Épocas de Treinamento: %d\n", network.config.numEpochs)
	fmt.Printf("Taxa de Aprendizado: %f\n", network.config.learningRate)
	fmt.Println("Pesos e Vieses da Camada Oculta:")
	ShowWeightsAndBiases(network.weightHidden, network.biasHidden)
	fmt.Println("Pesos e Vieses da Camada de Saída:")
	ShowWeightsAndBiases(network.weightOut, network.biasOut)
	fmt.Println(string(colorReset))
}

// ShowPredictedInfo exibe informações sobre a rede neural após a função predict().
func ShowPredictedInfo(network *neuralNet, predictions *mat.Dense) {
	colorReset := "\033[0m"
	colorGreen := "\033[32m"

	fmt.Println(string(colorGreen))
	fmt.Println("Informações da Rede Neural APÓS O Predict():")
	fmt.Printf("Arquitetura da Rede: %d -> %d -> %d\n", network.config.inputNeurons, network.config.hiddenNeurons, network.config.outputNeurons)
	fmt.Printf("Número de Épocas de Treinamento: %d\n", network.config.numEpochs)
	fmt.Printf("Taxa de Aprendizado: %f\n", network.config.learningRate)
	fmt.Println("Pesos e Vieses da Camada Oculta:")
	ShowWeightsAndBiases(network.weightHidden, network.biasHidden)
	fmt.Println("Pesos e Vieses da Camada de Saída:")
	ShowWeightsAndBiases(network.weightOut, network.biasOut)
	fmt.Println(string(colorReset))
}

// ShowWeightsAndBiases exibe os pesos e vieses de uma camada da rede neural.
func ShowWeightsAndBiases(weights, biases *mat.Dense) {
	fmt.Println("Pesos:")
	fmt.Println(mat.Formatted(weights, mat.Squeeze()))
	fmt.Println("Vieses:")
	fmt.Println(mat.Formatted(biases, mat.Squeeze()))
}
