#include "src/layer.h"
#include "src/network.h"
#include <string>
#include <fstream>

void createRandomMatrix(int rows, int cols, Matrix &matrix);
void saveMatrixToFile(const Matrix &matrix, int rows, int cols, std::string fileName);
void loadMatrixFromFile(Matrix &matrix, std::string fileName);
void createTestCasesForLayer(int numCases, int rows, int cols, std::string testDirRootName, Layer* layer);
void runTestCasesForLayer(int numCases, int rows, int cols, std::string testDirRootName, Layer* layer);