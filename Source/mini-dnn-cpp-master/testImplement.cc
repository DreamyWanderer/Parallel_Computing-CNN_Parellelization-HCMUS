#include "testImplement.h"

void createRandomMatrix(int rows, int cols, Matrix &matrix) {
  // Create random matrix with rows and cols, from range 0 to 255
    matrix.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j++) {
        matrix(i, j) = rand() % 256;
      }
    }
}

void saveMatrixToFile(const Matrix &matrix, int rows, int cols, std::string fileName) {
  // Save matrix to file
    std::ofstream file(fileName);
    if (file.is_open()) {
      file << rows << " " << cols << "\n";
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j++) {
          file << matrix(i, j) << " ";
        }
        file << "\n";
      }
    }
    else {
      std::cout << "Unable to open file " << fileName << std::endl;
    }
}

void loadMatrixFromFile(Matrix &matrix, std::string fileName) {
  // Load matrix from file
    std::ifstream file(fileName);
    if (file.is_open()) {
      int rows, cols;
      file >> rows >> cols;
      matrix.resize(rows, cols);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j++) {
          file >> matrix(i, j);
        }
      }
    }
}

void createTestCasesForLayer(int numCases, int rows, int cols, std::string testDirRootName, Layer* layer)
{
        for (int i = 0; i < numCases; ++i) {
            Matrix input;
            createRandomMatrix(rows, cols, input);
            layer->forward(input);
            Matrix output = layer->output().col(0);
            saveMatrixToFile(input, rows, cols, testDirRootName + "/input_" + std::to_string(i) + ".txt");
            saveMatrixToFile(output, output.rows(), 1, testDirRootName + "/output_" + std::to_string(i) + ".txt");
        }
}

void runTestCasesForLayer(int numCases, int rows, int cols, std::string testDirRootName, Layer* layer)
{
        for (int i = 0; i < numCases; ++i) {
            Matrix input;
            loadMatrixFromFile(input, testDirRootName + "/input_" + std::to_string(i) + ".txt");
            layer->forward(input);
            Matrix output = layer->output().col(0);
            Matrix expectedOutput;
            loadMatrixFromFile(expectedOutput, testDirRootName + "/output_" + std::to_string(i) + ".txt");
            if (output.rows() != expectedOutput.rows()) {
                std::cout << "Test case " << i << " failed: output rows mismatch" << std::endl;
                return;
            }
            if (output.cols() != expectedOutput.cols()) {
                std::cout << "Test case " << i << " failed: output cols mismatch" << std::endl;
                return;
            }
            if (output.isApprox(expectedOutput)) {
                std::cout << "Test case " << i << " passed" << std::endl;
            }
            else {
                std::cout << "Test case " << i << " failed: output mismatch" << std::endl;
                return;
            }
        }
        std::cout << "Test cases passed" << std::endl;
}
