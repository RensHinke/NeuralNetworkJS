const {add, matMul, transpose} = require('simple-linalg');
const fs = require('fs');

class Perceptron {
    constructor(y_real, learning_rate) {
        this.w1 = 3;
        this.w2 = -3;
        this.bias = 5;
        this.y_real = y_real;
        this.learning_rate = learning_rate;
    }

    forward(x1, x2) {
        return this.stepActivation(this.w1 * x1 + this.w2 * x2 + this.bias);
    }

    backwardsPropogate(x1, x2) {
        let loss = this.y_real[x1][x2] - this.forward(x1, x2)
        console.log(x1, x2, loss, this.y_real[x1][x2], this.forward(x1, x2))
        this.w1 += this.learning_rate * x1 * loss
        this.w2 += this.learning_rate * x2 * loss
        this.bias += this.learning_rate * loss
    }

    stepActivation(x) {
        if (x >= 0) {
            return 1;
        } else {
            return 0;
        }
    }
}

class ANN {
    constructor(neuronsPerLayer, initialWeights = null, initialBiases = null) {
        this.neuronsPerLayer = neuronsPerLayer;

        if (initialWeights == null) {
            this.weights = []
            for (let i = 0, n = neuronsPerLayer.length - 1; i < n; i++) {
                this.weights.push(create2dArrayUniform(neuronsPerLayer[i + 1], neuronsPerLayer[i], 2));
            }
        } else {
            this.weights = initialWeights;
        }

        if (initialBiases == null) {
            this.biases = new Array(neuronsPerLayer.length - 1);
            for (let i = 0, n = this.biases.length; i < n; i++) {
                this.biases[i] = new Array(neuronsPerLayer[i + 1]);
            }

            for (let i = 0, n = this.biases.length; i < n; i++) {
                for (let j = 0; j < this.biases[i].length; j++) {
                    this.biases[i][j] = uniform(8);
                }
            }
        } else {
            this.biases = initialBiases;
        }
    }

    feedForwardOneLayer(inputs, interLayer) {
        let a = add(matMul(this.weights[interLayer], transpose([inputs])), transpose([this.biases[interLayer]]));
        applyToAllElementsOfArray(a, sigmoid);
        return a;
    }

    feedForward(inputLayerInputs) {
        let results = new Array();
        let a = this.feedForwardOneLayer(inputLayerInputs, 0);
        results.push(a);
        
        if (this.neuronsPerLayer.length > 2) {
            for (let i = 1; i < this.neuronsPerLayer.length - 1; i++) {
                a = this.feedForwardOneLayer(a, i);
                
                results.push(a);
            }
        }
        return results;
    }

    train(inputLayerInputs, y, learningRate = 0.02) {
        let results = this.feedForward(inputLayerInputs);

        // Backpropagation
        let error_terms = new Array(this.neuronsPerLayer.length - 1);
        for (let i = 0; i < error_terms.length; i++) {
            error_terms[i] = new Array(this.neuronsPerLayer[i + 1]);
        }

        for (let i = 0; i < error_terms.length; i++) {
            for (let j = 0, n = error_terms[i].length; j < n; j++) {
                error_terms[i][j] = 0;
            }
        }

        for (let layer = error_terms.length - 1; layer >= 0; layer--) {
            for (let currentNeuron = 0, n = this.neuronsPerLayer[layer + 1]; currentNeuron < n; currentNeuron++) {
                    
                    error_terms[layer][currentNeuron] = results[layer][currentNeuron] * (1 - results[layer][currentNeuron]);
                    
                    if (layer == error_terms.length - 1) {
                        error_terms[layer][currentNeuron] *= (y[currentNeuron] - results[layer][currentNeuron]);
                    } else {
                        let sum = 0;
                        
                        for (let previousLayerNeuron = 0, n = error_terms[layer + 1].length; previousLayerNeuron < n; previousLayerNeuron++) {
                            sum += error_terms[layer + 1][previousLayerNeuron] * this.weights[layer + 1][previousLayerNeuron][currentNeuron];
                        }
                        error_terms[layer][currentNeuron] = sum;
                    }
            }
        }
        

        for (let layer = 0; layer < this.neuronsPerLayer.length - 1; layer++) {
            for (let currentNeuron = 0; currentNeuron < this.neuronsPerLayer[layer]; currentNeuron++) {
                for (let nextLayerNeuron = 0; nextLayerNeuron < this.neuronsPerLayer[layer + 1]; nextLayerNeuron++) {
                    if (layer == 0) {
                        this.weights[layer][nextLayerNeuron][currentNeuron] += error_terms[layer][nextLayerNeuron] * inputLayerInputs[currentNeuron] * learningRate;
                    } else {
                        this.weights[layer][nextLayerNeuron][currentNeuron] += error_terms[layer][nextLayerNeuron] * results[layer - 1][currentNeuron] * learningRate;
                    }
                }
            }
        }

        for (let layer = 0; layer < this.biases.length; layer++) {
            for (let currentNeuron = 0; currentNeuron < this.biases[layer].length; currentNeuron++) {
                this.biases[layer][currentNeuron] += error_terms[layer][currentNeuron] * learningRate;
            }
        }
    }
}

function main(data, targets, testData, testTargets, differentTargets) {
    let ann = new ANN([data[0].length, 50, differentTargets]);

    console.log("data.length == targetData.length", data.length == targets.length);
    console.log(data.length, targets.length);
    for (let i = 0; i < data.length; i++) {
        let y = new Array(differentTargets);
        y.fill(0);
        y[targets[i]] = 1;
        applyToAllElementsOf1dArray(data[i], devideBy255);
        ann.train(data[i], y, 0.1);
        if (i % 1000 == 0)
            console.log("i train", i);
    }
    let amountOfCorrect = 0;
    for (let testIndex = 0; testIndex < testData.length; testIndex++) {
        let y = new Array(differentTargets);
        y.fill(0);
        y[testTargets[testIndex]] = 1;
        applyToAllElementsOf1dArray(testData[testIndex], devideBy255);
        let prediction = indexOfMax(ann.feedForward(testData[testIndex]).pop());
        if (prediction == testTargets[testIndex]) {
            amountOfCorrect++;
        }
    }

    fs.writeFile('weights.txt', JSON.stringify(ann.weights), (err) => {

        // In case of a error throw err.
        if (err) throw err;
    })

    fs.writeFile('biases.txt', JSON.stringify(ann.biases), (err) => {

        // In case of a error throw err.
        if (err) throw err;
    })

    console.log((amountOfCorrect / testData.length) * 100 + "%");
}

function indexOfMax(arr) {
    return arr.reduce((maxIndex, elem, i, arr) => 
        elem > arr[maxIndex] ? i : maxIndex, 0);
} 

function uniform(range) {
    return Math.random() * range - range / 2;
}

function create2dArrayUniform(rows, columns, range) {
    let array = Array(rows).fill().map(() => Array(columns).fill(0));
    for (let i = 0; i < array.length; i++) {
        for (let j = 0; j < array[i].length; j++) {
            array[i][j] = uniform(range);
        }
    }
    return array;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function sigmoidDerivative(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

function applyToAllElementsOfArray(array, f) {
    for (let i = 0; i < array.length; i++) {
        for (let j = 0; j < array[i].length; j++) {
            array[i][j] = f(array[i][j]);
        }
    }
}

function applyToAllElementsOf1dArray(array, f) {
    for (let i = 0; i < array.length; i++) {
        array[i] = f(array[i]);
    }
}

function devideBy255(x) {
    return x / 255;
}